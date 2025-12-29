import torch
from .base_pipeline.base_model import MatchingBaseModel
from .attention_layers.cross_attention import CrossAttentionLayer
from .attention_layers.point_transformer import PointTransformer
from .feature_extractor import build_feature_extractor
from .surface_segmentation.segmentation_classifier import SegmentationClassifier
from .multipart_matching.affinity import AffinityDual
from .multipart_matching.utils_sinkhorn import Sinkhorn


class JoingSegmentationAlignmentModel(MatchingBaseModel):
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
        self.half_dim = self.aff_feat_dim // 2 # 256 for primal/dual split

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
        """
        sinkhorn = Sinkhorn(
            max_iter=self.config.MODEL.SINKHORN_MAXITER, # default 20
            tau=self.config.MODEL.SINKHORN_TAU # default 0.05
        )
        return sinkhorn