import torch
from .base_pipeline.base_model import MatchingBaseModel
from .attention_layers.cross_attention import CrossAttentionLayer
from .attention_layers.point_transformer import PointTransformer
from .feature_extractor import build_feature_extractor
from .surface_segmentation.segmentation_classifier import SegmentationClassifier
from .multipart_matching.affinity import AffinityDual
from .multipart_matching.utils_sinkhorn import Sinkhorn
from .multipart_matching.utils_hungarian import hungarian
import torch.nn as nn
import torch.nn.functional as fun
from multipart_matching.utils_matching import get_fracture_points_from_label
from .utils import get_batch_length_from_part_points
from surface_segmentation.segmentation_classifier import compute_label
from surface_segmentation.utils import diagonal_square_matrix


class JointSegmentationAlignmentModel(MatchingBaseModel):
    """
    Jigsaw Model: joint learning of segmentation, matching and alignment.

    Implements the full Jigsaw pipeline:
    1. Front-end feature extractor with self/cross-attention
    2. Surface segmentation module
    3. Multi-part matching module with primal-dual descriptor
    4. Global fracture alignment module
    """

    def __init__(self, config):
        """
        Configure model with initial hyperparameters from config.
        """
        super().__init__(config)

        # Segmentation parameters
        self.num_classes = 2  # binary segmentation: fracture vs non-fracture
        self.pc_cls_method = self.config.MODEL.PC_CLS_METHOD.lower() # "binary" or "multi-class"
        self.num_classes = self.config.MODEL.PC_NUM_CLS

        # Affinity parameters
        self.aff_feat_dim = self.config.MODEL.AFF_FEAT_DIM # default 512
        assert self.aff_feat_dim % 2 == 0, "Affinity feature dimension must be even for primal-dual splitting."
        self.half_aff_feat_dim = self.aff_feat_dim // 2 # 256 for primal/dual split

        # Loss weights
        self.w_cls_loss = self.config.MODEL.LOSS.w_cls_loss # segmentation loss weight: α, always 1.0
        self.w_mat_loss = self.config.MODEL.LOSS.w_mat_loss # matching loss weight:  β, starts at 0, becomes 1.0 at epoch 9
        self.w_rig_loss = self.config.MODEL.LOSS.w_rig_loss # rigid alignment loss weight: γ, starts at 0, becomes 1.0 at epoch 199

        # Attention layers
        # Self-attention layer: aggregate local features within each piece
        self.self_attention = PointTransformer(
            in_features=self.pc_feat_dim,
            out_features=self.pc_feat_dim,
            n_heads=self.config.MODEL.TF_NUM_HEADS,
            k_neighbors=self.config.MODEL.TF_NUM_SAMPLE
        )
        # Cross-attention layer: exchange features across pieces
        self.cross_attention = CrossAttentionLayer(
            n_head=self.config.MODEL.TF_NUM_HEADS,
            d_input=self.pc_feat_dim,
        )
        self.attention_layers = [("self", self.self_attention), ("cross", self.cross_attention)]

        # Initialize model components
        self.feature_extractor = self._init_feature_extractor() # PointNet++ based feature extractor
        self.segmentation_classifier = self._init_segmentation_classifier() # fracture surface segmentation head
        self.affinity_extractor = self._init_affinity_extractor() # affinity feature projection head
        self.affinity_layer = self._init_affinity_layer() # primal-dual affinity layer
        self.sinkhorn = self._init_sinkhorn() # differentiable optimal transport layer


    def _init_feature_extractor(self):
        """
        Initialize point cloud feature extractor.

        Output:
            feature_extractor (nn.Module): point cloud feature extractor model
        """
        feature_extractor = build_feature_extractor(
            self.config.MODEL.ENCODER,
            features_dimension=self.pc_feat_dim,
            global_feat=False,
            in_feat_dim=3 # input 3D coordinates (x, y, z)
        )
        return feature_extractor

    def _init_segmentation_classifier(self):
        """
        Initialize point-wise segmentation classifier.

        Output:
            segmentation_classifier (nn.Module): point cloud segmentation classifier
        """
        segmentation_classifier = SegmentationClassifier(
            model_point=self.pc_cls_method,
            pc_feat_dim=self.pc_feat_dim,
            num_classes=self.num_classes
        )
        return segmentation_classifier

    def _init_affinity_extractor(self):
        """
        Turn point features into affinity features for primal-dual matching.

        Output:
            affinity_extractor (nn.Module): affinity feature projection head
        """
        affinity_extractor = nn.Sequential(
            nn.BatchNorm1d(self.pc_feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.pc_feat_dim, self.aff_feat_dim, kernel_size=1) # 1x1 convolution = MLP
        )
        return affinity_extractor

    def _init_affinity_layer(self):
        """
        Turn affinity features into primal-dual affinity scores (how likely two pieces match).

        Output:
            affinity_layer (nn.Module): primal-dual affinity layer
        """
        affinity_layer = AffinityDual(feature_dim=self.aff_feat_dim)
        return affinity_layer

    def _init_sinkhorn(self):
        """
        Initialize Sinkhorn layer for differentiable optimal transport.
        Output:
            sinkhorn (nn.Module): Sinkhorn optimal transport layer
        """
        sinkhorn = Sinkhorn(
            max_iter=self.config.MODEL.SINKHORN_MAXITER, # default 20
            tau=self.config.MODEL.SINKHORN_TAU # default 0.05
        )
        return sinkhorn

    def _extract_part_features(self, part_pcs, batch_length):
        """
        Extract point features for all parts in the batch.

        Input:
            part_pcs: [B, N_SUM, 3] - concatenated point clouds of all pieces in the batch
            batch_length: [sum(n_valid)] - lengths of all valid pieces in the batch

        Output:
            part_features: [B, N_SUM, F] - extracted point features for all pieces
        """
        B, N_SUM, _ = part_pcs.shape

        # we flatten the batch dimension because we have shared-weight across all pieces
        valid_pcs = part_pcs.reshape(B * N_SUM, -1)
        part_features = self.feature_extractor(valid_pcs, batch_length)  # [B * N_SUM, F]

        part_features = part_features.reshape(B, N_SUM, -1)  # [B, N_SUM, F]
        return part_features

    def _get_critical_features_from_label(self, B, N, F, features, n_critical_pcs, critical_labels):
        """
        Extract features of critical fracture points based on ground truth/predicted labels.
        Input:
            B: int - batch size
            N: int - number of points in concatenated point clouds
            F: int - feature dimension
            features: [B, N, F] - point features for all points
            n_critical_pcs: [B] - number of critical fracture points in each batch
            critical_labels: [B, N] - binary labels for each point (1: critical fracture point, 0: non-critical)

        Output:
            critical_features: [B, N, F] - features of critical fracture points (zero-padded)
        """
        critical_features = torch.zeros(B, N, F, device=self.device, dtype=features.dtype)
        
        for b in range(B):
            critical_features[b, :n_critical_pcs[b]] = features[b, critical_labels[b] == 1]

        return critical_features

    def forward(self, data_dict):
        """
        Forward pass implementing the complete Jigsaw pipeline.

        This is the main forward pass that executes all stages:
        1. Feature extraction from point clouds (Section 3.1)
        2. Self/cross-attention for feature aggregation (Section 3.1)
        3. Fracture surface segmentation (Section 3.2)
        4. Multi-part matching with primal-dual descriptor (Section 3.3)
        
        Input:
            data_dict: dict - input data dictionary containing:
                'part_pcs': [B, N_SUM, 3] - concatenated point clouds of all pieces in the batch
                'n_pcs': [B, P] - number of points in each piece
                'part_valids': [B, P] - binary indicators for valid parts in each pair
                'gt_pcs': [B, N_SUM, 3] - ground truth concatenated point clouds (for training)
                'part_feats': [B, N_SUM, F] - cached point features 
                'critical_pcs_idx': [B, N_SUM] - indices of critical fracture points 
                'n_critical_pcs': [B, P] - number of critical fracture points per piece

        Output:
            out_dict: dict - output data dictionary containing:
                'cls_logits': [B, N_SUM, C] - segmentation logits for each point
                'cls_preds': [B, N_SUM] - segmentation predictions for each point
                'ds_mat': [B, N_CRIT_MAX, N_CRIT_MAX] - doubly stochastic matching matrix
                'perm_mat': [B, N_CRIT_MAX, N_CRIT_MAX] - discrete matching matrix (during testing)
                'batch_size': int - batch size
        """
        out_dict = {}

        valid_parts = data_dict['part_valids']  # (B, 2) binary indicators for valid parts in each pair
        n_valid = torch.sum(valid_parts, dim=1).to(torch.long)  # [B] number of valid pieces
        
        part_pcs = data_dict['part_pcs'] # [B, N_SUM, 3] concatenated point clouds of all pieces in the batch
        n_pcs = data_dict['n_pcs']  # [B,P] number of points in each piece

        B, N_SUM, _ = part_pcs.shape
        part_features = data_dict.get('part_feats', None) # [B, N_SUM, F] cached features if available
        batch_length = get_batch_length_from_part_points(n_pcs, n_valid).to(self.device) # [sum(n_valid)] lengths of all valid pieces in the batch

        # STEP 1: extract point features
        if part_features is None:
            part_features = self._extract_part_features(part_pcs, batch_length)  # [B, N_SUM, F]
            
            # apply self-attention and cross-attention layers
            for name, layer in self.attention_layers:
                if name == "self":
                    # self attention: aggregate local features within each piece
                    part_features = layer(
                            part_pcs.reshape(-1, 3).contiguous(), # point transformer expects a (flat) point cloud input
                            part_features.view(-1, self.pc_feat_dim), # we flatten this too because coordinates and features must be aligned
                            batch_length
                        )
                        .view(B, N_SUM, -1) # reshape back to (B, N_SUM, F)
                        .contiguous()
                elif name == "cross":
                    # cross attention: propagate info between pieces
                    part_features = layer(part_features)  # [B, N_SUM, F]
            
            data_dict.update({'part_feats': part_features}) # cache extracted features for later reuse
        

        # STEP 2: point cloud segmentation
        segmentation_features = part_features.transpose(1, 2) # [B, F, N_SUM] for point-wise classification

        # compute segmentation logits and predictions
        cls_logits = self.segmentation_classifier(segmentation_features)  # [B, C, N_SUM]
        cls_logits = cls_logits.permute(0, 2, 1).contiguous()  # [B, N_SUM, C]

        with torch.no_grad(): # no gradient for predictions -> no learning signal
            if self.pc_cls_method == "binary":
                # we apply sigmoid for binary classification
                probs = torch.sigmoid(cls_logits)  # [B, N_SUM, 1]
                cls_preds = (probs > 0.5).long()
            else:  # multi-class
                # we do not apply softmax here since argmax is invariant to monotonic transformations
                cls_preds = torch.argmax(cls_logits, dim=-1)  # [B, N_SUM]

        out_dict.update({
            'cls_logits': cls_logits,
            'cls_preds': cls_preds,
            'batch_size': B
        })

        if not self.training and self.trainer.testing: # if testing phase
            # during inference we use the predicted segmentation to determine critical fracture points
            # because we do not have ground truth geometry available
            with torch.no_grad():
                fracture_preds = cls_preds
                n_critical_pcs, critical_pcs_idx = get_fracture_points_from_label(n_pcs, fracture_preds)
                data_dict.update({
                    'n_critical_pcs': n_critical_pcs,
                    'critical_pcs_idx': critical_pcs_idx,
                    'critical_label': fracture_preds
                })

        else:
            # during training we know the ground truth positions and we can determine from the geometry
            # of the pieces which points are critical fracture points (points close to other pieces)
            if 'n_critical_pcs' not in data_dict:
                with torch.no_grad():
                    gt_pcs = data_dict['gt_pcs'] # [B, N_SUM, 3]
                    critical_threshold = data_dict['critical_label_threshold'] 

                    fracture_preds = compute_label(points=gt_pcs, nr_points_piece=n_pcs, nr_valid_pieces=n_valid, dist_thresholds=critical_threshold)
                    n_critical_pcs, critical_pcs_idx = get_fracture_points_from_label(n_pcs, fracture_preds)
                    data_dict.update({
                        'n_critical_pcs': n_critical_pcs,
                        'critical_pcs_idx': critical_pcs_idx,
                        'critical_label': fracture_preds
                    })
            else:
                fracture_preds = data_dict['critical_label']
            
            # early return during warm-up phase (before matching loss is activated)
            if self.w_mat_loss == 0:
                # only train segmentation in early epochs (before epoch 9) to stabilize segmentation training
                return out_dict


        # STEP 3: multi-part matching
        feature_dim = part_features.shape[-1]
        n_critical_pcs_object = torch.sum(n_critical_pcs, dim=-1) # [B] number of critical fracture points in each object
        n_critical_max = torch.max(n_critical_pcs_object) # max number of critical points in the batch

        # extract features of critical fracture points
        critical_features = self._get_critical_features_from_label(B, n_critical_max, feature_dim, part_features, n_critical_pcs, fracture_preds)  # [B, N_CRIT_MAX, F]
       
        # project to affinity feature space
        matching_descriptors = self.affinity_extractor(critical_features.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N_CRIT_MAX, AFF_F]
        matching_descriptors = torch.cat( 
            # normalize primal and dual features separately
            [
                fun.normalize(matching_descriptors[:, :, :self.half_aff_feat_dim], p=2, dim=-1),  # primal features
                fun.normalize(matching_descriptors[:, :, self.half_aff_feat_dim:], p=2, dim=-1)   # dual features
            ],
            dim=-1
        )

        # compute affinity matrix using primal-dual matching
        # M = primal(X) * A * dual(Y)^T
        affinity_scores = self.affinity_layer(matching_descriptors, matching_descriptors)  # [B, N_CRIT_MAX, N_CRIT_MAX] primal-dual affinity scores

        # mask self-matching during testing
        if not self.training and self.config.MODEL.TEST_S_MASK:
            mask = diagonal_square_matrix(affinity_scores.shape, nr_points_piece=n_critical_pcs, nr_valid_pieces=n_valid, pos_msk=1, neg_msk=0).detach()
            neg_mask = diagonal_square_matrix(affinity_scores.shape, nr_points_piece=n_critical_pcs, nr_valid_pieces=n_valid, pos_msk=0, neg_msk=-1e6).detach()
            masked_affinity_scores = affinity_scores * mask + neg_mask
            out_dict.update({
                's_mask': mask,
                's_neg_mask': neg_mask,
            })
        else:
            masked_affinity_scores = affinity_scores

        # normalize affinity matrix to doubly stochastic matrix using Sinkhorn
        soft_matching_matrix = self.sinkhorn(masked_affinity_scores, nrows=n_critical_pcs_object, ncols=n_critical_pcs_object)  # [B, N_CRIT_MAX, N_CRIT_MAX] normalized matching matrix
        out_dict.update({
            'ds_mat': soft_matching_matrix, # doubly stochastic matching matrix
        })

        if not self.training:
            # during testing we compute discrete matching using Hungarian algorithm
            hard_matching_matrix = hungarian(soft_matching_matrix, n_critical_pcs_object, n_critical_pcs_object)  # [B, N_CRIT_MAX, N_CRIT_MAX] discrete matching matrix
            out_dict.update({
                'perm_mat': hard_matching_matrix, # hard matching matrix
            })


        return out_dict

    def _loss_function(self, data_dict, out_dict, optimizer_idx=-1):
        """
        Compute the complete loss function for training.
        This is: L = α * L_seg + β * L_mat + γ * L_rig
        
        - L_seg: segmentation loss (binary cross-entropy)
        - L_mat: matching loss (cross-entropy on matching matrix)
        - L_rig: rigid alignment loss (Chamfer distance after alignment)
        - α = 1.0 always
        - β = 0 for epochs < 9 and 1.0 afterwards (allows segmentation to converge first)
        - γ = 0 for epochs < 199 and 1.0 afterwards (applied at the end for refinement)

        Input:
            data_dict: input data dictionary
            out_dict: output data dictionary from forward pass
            optimizer_idx: int - index of the optimizer (for multi-optimizer setups)

        Output:
            loss_dict
        """
        loss_dict = {}

        gt_pcs = data_dict['gt_pcs']  # [B, N_SUM, 3] ground truth concatenated point clouds
        part_pcs = data_dict['part_pcs']  # [B, N_SUM, 3] input concatenated point clouds
        n_pcs = data_dict['n_pcs']  # [B, P] number of points in each piece
        
        valid_parts = data_dict['part_valids']  # (B, 2) binary indicators for valid parts in each pair
        n_valid = torch.sum(valid_parts, dim=1).to(torch.int)  # [B] number of valid pieces
        
        n_critical_pcs = data_dict['n_critical_pcs']  # [B, P] number of critical fracture points per piece
        n_critical_pcs_object = torch.sum(n_critical_pcs, dim=-1)  # [B] number of critical fracture points in each object
        n_critical_max = torch.max(n_critical_pcs_object, dim=-1)  # maximum number of critical fracture points in the batch
        critical_pcs_idx = data_dict['critical_pcs_idx']  # [B, N_SUM] indices of critical fracture points
        critical_labels = data_dict['critical_label']  # [B, N_SUM] binary labels for each point (1: critical fracture point, 0: non-critical)

        cls_logits = out_dict['cls_logits']  # [B, N_SUM, C] segmentation logits
        cls_preds = out_dict['cls_preds']  # [B, N_SUM] segmentation predictions

        B, N_SUM, _ = part_pcs.shape
        loss_dict.update({
            'batch_size': B
        })


        # SEGMENTATION LOSS
        seg_gt = critical_labels.reshape(-1) # critical_labels is a binary fracture mask for all points, after reshaping we get [B * N_SUM]
        
        if self.pc_cls_method == "binary":
            # binary cross-entropy for fracture vs non-fracture
            cls_logits = cls_logits.reshape(-1)
            cls_loss = fun.binary_cross_entropy_with_logits(cls_logits, seg_gt.to(torch.float32))
        else:
            # multi-class cross-entropy for multiple fracture classes
            cls_logits = cls_logits.reshape(-1, self.num_classes) 
            cls_loss = fun.nll_loss(cls_logits, seg_gt) # negative log-likelihood loss

        cls_preds = cls_preds.reshape(-1)

        cls_accuracy = torchmetrics.functional.accuracy(cls_preds, seg_gt, task="binary")
        cls_precision = torchmetrics.functional.precision(cls_preds, seg_gt, task="binary")
        cls_recall = torchmetrics.functional.recall(cls_preds, seg_gt, task="binary")
        cls_f1_score = torchmetrics.functional.f1_score(cls_preds, seg_gt, task="binary")

        loss_dict.update({
            'cls_loss': cls_loss,
            'cls_acc': cls_accuracy,
            'cls_precision': cls_precision,
            'cls_recall': cls_recall,
            'cls_f1': cls_f1_score
        })

        if self.training and self.w_mat_loss == 0:
            # early return during warm-up phase (before matching loss is activated)
            loss_dict.update({"loss": cls_loss})
            return loss_dict


        # MATCHING LOSS
        # compute ground truth matching matrix based on nearest neighbors
        with torch.no_grad():
            # extract ground truth positions for fracture points
            gt_fracture_xyz = self._get_critical_features_from_label(B, n_critical_max, 3, gt_pcs, n_critical_pcs_object, critical_labels)
            # compute pairwise distances between ground truth fracture points
            gt_pairwise_distances = square_distance(gt_fracture_xyz, gt_fracture_xyz)  # [B, N_CRIT_MAX, N_CRIT_MAX]

            mask = out_dict.get('s_mask', None)
            neg_mask = out_dict.get('s_neg_mask', None)
            if mask is None:
                mask = diagonal_square_matrix(shape=(B, n_critical_max, n_critical_max), nr_points_piece=n_critical_pcs, nr_valid_pieces=n_valid, pos_msk=1, neg_msk=0)
            if neg_mask is None:
                neg_mask = diagonal_square_matrix(shape=(B, n_critical_max, n_critical_max), nr_points_piece=n_critical_pcs, nr_valid_pieces=n_valid, pos_msk=0, neg_msk=-1e6)
            
            gt_pairwise_distances -= neg_mask # mask out self-matching distances
            # we pick the nearest neighbor in other pieces
            gt_nn_indices = torch.argmin(gt_pairwise_distances, dim=-1).reshape(B, n_critical_max, -1)  # [B, N_CRIT_MAX]
        
            # create binary matching matrix
            gt_matching_matrix = torch.zeros(B, n_critical_max, n_critical_max, device=self.device).scatter_(2, gt_nn_indices, 1)  # [B, N_CRIT_MAX, N_CRIT_MAX]
            gt_matching_matrix *= mask # apply mask to remove self-matching

        ds_mat = out_dict['ds_mat'] # predicted doubly stochastic matching matrix
        mat_loss = permutation_loss(ds_mat, gt_matching_matrix, n_critical_pcs_object, n_critical_pcs_object)
        loss_dict.update({
            'mat_loss': mat_loss,
            'n_critical_max': n_critical_max
        })


    # RIGIDITY LOSS
    if self.w_rig_loss > 0:
        rig_loss = rigidity_loss(n_pcs, n_valid, gt_pcs, part_pcs, n_critical_pcs, critical_pcs_idx, ds_mat)
        loss_dict.update({
            'rig_loss': rig_loss
        })
    else:
        rig_loss = 0


    # TOTAL WEIGHTED LOSS
    if self.training:
        # loss = α * L_seg + β * L_mat + γ * L_rig
        loss = self.w_cls_loss * cls_loss + self.w_mat_loss * mat_loss + self.w_rig_loss * rig_loss
    else:
        # during validation we use unweighted sum
        loss = cls_loss + mat_loss

    loss_dict.update({
        'loss': loss
    })

    # compute matching metrics to monitor training progress
    perm_mat = out_dict.get('perm_mat', None) # discrete matching matrix (only during testing)
    if perm_mat is not None:
        # compute precision, recall, F1-score for matching
        tp, fp, fn = 0, 0, 0
        for b in range(B):
            num_valid_points = n_critical_pcs_object[b]
            pred_matching_matrix = perm_mat[b, :num_valid_points, :num_valid_points]
            gt_matching_matrix_b = gt_matching_matrix[b, :num_valid_points, :num_valid_points]

            # true positive: predicted match AND GT match
            tp += torch.sum(pred_matching_matrix * gt_matching_matrix_b).float()
            # false positive: predicted match BUT NOT GT match
            fp += torch.sum(pred_matching_matrix * (1 - gt_matching_matrix_b)).float()
            # false negative: NOT predicted match BUT GT match
            fn += torch.sum((1 - pred_matching_matrix) * gt_matching_matrix_b).float()

        const = torch.tensor(1e-7, device=self.device)
        precision = tp / (tp + fp + const)
        recall = tp / (tp + fn + const)
        f1_score = 2 * precision * recall / (precision + recall + const)

        loss_dict.update({
            'mat_precision': precision,
            'mat_recall': recall,
            'mat_f1': f1_score
        })

    return loss_dict

    def training_epoch_end(self, outputs):
        """
        Callback at the end of each training epoch.
        It implements loss weight scheduling:
        - matching loss enabled at epoch 9
        - rigidity loss enabled at epoch 199

        This scheduling allows:
        1. segmentation to converge first (epochs 0-8)
        2. them matching is added (epochs 9-198)
        3. finally rigidity loss is added for refinement (epoch 199+)
        """
        # enable matching loss after warm-up period
        if self.w_mat_loss == 0 and self.current_epoch >= self.config.MODEL.LOSS.mat_epoch:
            self.w_mat_loss = 1.0
            print(f"Epoch {self.current_epoch}: Matching loss activated.")

        # enable rigidity loss for final refinement
        if self.w_rig_loss == 0 and self.current_epoch >= self.config.MODEL.LOSS.rig_epoch:
            self.w_rig_loss = 1.0
            print(f"Epoch {self.current_epoch}: Rigidity loss activated.")