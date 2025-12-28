import os
import pickle
import torch
import pytorch_lightning
import torch.optim as optim
import numpy as np

from utilities.global_alignment import estimate_global_transform
from utilities.optimization import filter_weight_decay_params

from utilities.rotation import Rotation3D
from .utils_evaluation import trans_metric, rot_metric, part_acc_and_cd
from .utils_lr_scheduler import CosineAnnealingWarmupRestarts
from .utils import dict_to_numpy


class MatchingBaseModel(pytorch_lightning.LightningModule):
    """
    Base class for fracture assembly models using PyTorch Lightning.
    Provides infrastructure for training, evaluation, and global alignment.
    """
    def __init__(self, config):
        super(MatchingBaseModel, self).__init__()
        self.config = config

        self.save_hyperparameters() # by default saves all init args to checkpoint
        self.max_num_part = self.config.DATA.MAX_NUM_PART # max nr of pieces per object
        self.part_comp_feat_dim = self.config.MODEL.PC_FEAT_DIM # feature dimension (default: 512)

        self.test_results = None
        if len(self.config.STATS):
            os.makedirs(self.config.STATS, exists_ok=True)
            self.stats = dict()
            self.stats['datas'] = [] # ground truth data
            self.stats['preds'] = [] # predicted transformations
            self.stats['metrics'] = [] # computed metrics
        else:
            self.stats = None

    def forward(self, data_dict):
        """
        Forward pass to predict segmentation & matching.
        To be implemented in subclass JoingSegmentationAlignmentModel.

        Input:
            data_dict: dict - input data containing point clouds and other info
        Output:
            output_dict: dict - Dictionary with predictions (cls_pred, ds_mat, perm_mat)
        """
        raise NotImplementedError("Forward method must be implemented per model.")

    def training_step(self, data_dict: dict, batch_idx: int, optimizer_idx: int = -1):
        """
        PyTorch Lightning training step.
        Called for each training batch; it computes forward pass and loss.

        Input:
            data_dict: batch data from dataloader
            batch_idx: index of the current batch
            optimizer_idx: index of the optimizer (if multiple optimizers are used)
        Output:
            loss: scalar loss for backpropagation
        """
        loss_dict = self.forward_pass(data_dict, mode='train', optimizer_idx=optimizer_idx)
        return loss_dict['loss']

    def validation_step(self, data_dict: dict, batch_idx: int, optimizer_idx: int = -1):
        """
        PyTorch Lightning validation step.
        Called for each validation batch; it computes losses and metrics.

        Input:
            data_dict: batch data from dataloader
            batch_idx: index of the current batch
            optimizer_idx: index of the optimizer (if multiple optimizers are used)
        Output:
            loss_dict: dict - dictionary with validation losses and metrics
        """
        loss_dict = self.forward_pass(data_dict, mode='val', optimizer_idx=optimizer_idx)
        return loss_dict

    def validation_epoch_end(self, outputs):
        """
        Aggregate validation results at the end of an epoch.
        Computes weighted average of losses across all validation batches,
        accounting for different batch sizes. This gives more accurate 
        average loss than simple mean.

        Input:
            outputs: list of loss_dict from each validation step
        """
        # handle both int and tensor batch sizes
        if isinstance(outputs[0]['batch_size'], int):
            func = torch.tensor
        else:
            func = torch.stack
        batch_sizes = func([out.pop('batch_size') for out in outputs]).type_as(outputs[0]['loss']) # [num_batches]

        # Collect losses by key
        losses = {
            f'val/{k}': torch.stack([out[k] for out in outputs]).reshape(-1)
            for k in outputs[0] if k != 'batch_size'
        }

        # Compute weighted averages
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum() for k, v in losses.items()
        }

        self.log_dict(avg_loss, sync_dist=True)

    def test_step(self, data_dict: dict, batch_idx: int, optimizer_idx: int = -1):
        """
        PyTorch Lightning test step.
        Called for each test batch; it performs forward pass, global alignment,
        and metric computation (PA, RE, TE, CD).

        Input:
            data_dict: batch data from dataloader
            batch_idx: index of the current batch
            optimizer_idx: index of the optimizer (if multiple optimizers are used)
        Output:
            loss_dict: dictionary with losses and metrics
        """
        loss_dict = self.forward_pass(data_dict, mode='test', optimizer_idx=optimizer_idx)
        return loss_dict

    def test_epoch_end(self, outputs):
        """
        Aggregate test results at the end of an epoch.
        Compute final evaluation metrics and saves stats.

        Input:
            outputs: list of loss_dict from each test step
        """
        # handle both int and tensor batch sizes
        if isinstance(outputs[0]['batch_size'], int):
            func_bs = torch.tensor
            func_loss = torch.stack
        else:
            func_bs = torch.cat
            func_loss = torch.cat
        batch_sizes = func_bs([out.pop('batch_size') for out in outputs]).type_as(outputs[0]['loss']) # [num_batches]

        # stack losses from all batches
        losses = {
            f'test/{k}': func_loss([out[k] for out in outputs])
            for k in outputs[0] if k != 'batch_size'
        }
        
        # compute weighted averages
        avg_loss = {
            k: (v * batch_sizes).sum() / batch_sizes.sum() for k, v in losses.items()
        }

        # print final metrics (PA, RE, TE, CD)
        print('; '.join([f'{k}: {v.item():.6f}' for k, v in avg_loss.items()]))

        # store results for external access
        self.test_results = avg_loss
        self.log_dict(avg_loss, sync_dist=True)

        if self.config.STATS is not None:
            with open(os.path.join(self.config.STATS, 'saved_stats.pk'), 'wb') as f:
                pickle.dump(self.stats, f)

    @torch.no_grad()
    def calc_metric(self, data_dict, trans_dict):
        """
        Compute evaluation metrics for fracture assembly.
        - Part Accuracy (PA): Percentage of correctly matched parts
        - Chamfer Distance (CD): Point cloud alignment quality (between predicted and GT point clouds)
        - Rotation metrics (RE): MSE, RMSE, MAE of rotation errors
        - Translation metrics (TE): MSE, RMSE, MAE of translation errors

        Input:
            data_dict: dictionary containing:
                - part_pcs: point clouds [B, P, 3]
                - part_quat or part_rot: ground truth rotations [B, P, 4] or [B, P, 3, 3]
                - part_trans: ground truth translations [B, P, 3]
                - part_valids: valid piece masks [B, P]
            trans_dict: dictionary containing:
                - pred_rot: predicted rotations [B, P, 3, 3]
                - trans: predicted translations [B, P, 3]
        Output:
            metrics_dict: dictionary with all evaluation metrics
        """
        # convert quaternions to rotation matrices if needed
        if 'part_rot' not in data_dict:
            part_quat = data_dict.pop('part_quat')  # [B, P, 4]
            data_dict['part_rot'] = Rotation3D(part_quat, rot_type='quat').convert('rmat')

        part_valids = data_dict['part_valids']
        part_pcs = data_dict['part_pcs']
        n_pcs = data_dict['n_pcs'] # [B] number of valid pieces per object
        metric_dict = dict()

        # convert predictions to tensors
        predicted_trans = torch.tensor(trans_dict['trans'], dtype=torch.float32, device=part_pcs.device)
        predicted_rot = torch.tensor(trans_dict['rot'], dtype=torch.float32, device=part_pcs.device)
        predicted_rot = Rotation3D(predicted_rot, rot_type='rmat')

        # ground truth values
        gt_trans = data_dict['part_trans']
        gt_rot = data_dict['part_rot']

        N_SUM = part_pcs.shape[1] # number of pieces per object
        B, P = n_pcs.shape

        # resample point clouds to have same nr of points per piece; easier CD computation
        part_pcs_resampled = []
        for b in range(B):
            piece_points_cumsum = 0
            resampled_pcs = []
            for p in range(P):
                if n_pcs[b, p].item() == 0: # how many points belong to part p of object b
                    # if padding piece: repeat last valid point
                    idx = torch.randint(low=piece_points_cumsum - 1, high=piece_points_cumsum, size=(N_SUM,))
                else:
                    # if valid piece: randomly sample N_SUM points from the piece
                    idx = torch.randint(low=piece_points_cumsum, high=piece_points_cumsum + n_pcs[b, p].item(), size=(N_SUM,))
                
                resampled_pcs.append(part_pcs[b, idx, :])
                piece_points_cumsum += n_pcs[b, p].item()
            
            resampled_pcs = torch.stack(resampled_pcs) # [P, N_SUM, 3]
            part_pcs_resampled.append(resampled_pcs)
        
        part_pcs_resampled = torch.stack(part_pcs_resampled).to(part_pcs.device) # [B, P, N_SUM, 3]

        # part accuracy and chamfer distance
        part_acc, cd = part_acc_and_cd(part_pcs_resampled, predicted_trans, gt_trans, predicted_rot, gt_rot, part_valids)
        metric_dict['part_acc'] = part_acc.mean()
        metric_dict['chamfer_distance'] = cd.mean()

        # rotation and translation errors: mse, rmse, mae
        for metric in ['mse', 'rmse', 'mae']:
            trans_met = trans_metric(predicted_trans, gt_trans, part_valids, metric)
            metric_dict[f'trans_{metric}'] = trans_met.mean()

            rot_met = rot_metric(predicted_rot, gt_rot, part_valids, metric)
            metric_dict[f'rot_{metric}'] = rot_met.mean()

        if self.stats is not None:
            saved_data = {
                'gt_trans': gt_trans,
                'gt_rot': gt_rot,
                'data_id': data_dict['data_id'],
            }
            self.stats['datas'].append(saved_data)
            self.stats['metrics'].append(dict_to_numpy(metric_dict))
            self.stats['preds'].append(dict_to_numpy(trans_dict))

        return metric_dict

    def _loss_function(self, data_dict, out_dict, optimizer_idx=-1):
        """
        Compute loss for fracture assembly.
        Implemented in subclass.

        Child class implements:
        - segmentation loss (L_seg)
        - matching loss (L_mat)
        - rigidity loss (L_rig)
        """
        raise NotImplementedError("Loss function must be implemented per model.")

    def global_alignment(self, data_dict, predictions):
        """
        Recover 6DOF poses for all pieces via global alignment.

        Given the pairwise matching matrix, this step:
        1. Constructs a pose graph with relative transformations
        2. Performs global synchronization to recover absolute poses
        3. Refines poses to minimize alignment error

        Input: 
            data_dict: input data dictionary
            predictions: model output with predicted matching matrices

        Output:
            pose_dict: dictionary with predicted rotations and translations
        """
        pred_pairwise_matches = predictions['perm_mat'].cpu().numpy()
        gt_object_pcs = data_dict['gt_pcs'].cpu().numpy()  # [B, M, 3]
        input_part_pcs = data_dict['part_pcs'].cpu().numpy()  # [B, N, 3]
        gt_part_rot = data_dict['part_quat'].cpu().numpy()  # [B, P, 3, 3]
        gt_part_trans = data_dict['part_trans'].cpu().numpy()  # [B, P, 3]
        
        n_pcs = data_dict.get('n_pcs', None)
        if n_pcs is not None:
            n_pcs = n_pcs.cpu().numpy()
        assert n_pcs is not None

        part_valids = data_dict.get('part_valids', None)
        n_valid = None
        if part_valids is not None:
            part_valids = part_valids.cpu().numpy()
            n_valid = np.sum(part_valids, axis=1, dtype=np.int32)  # [B]
        assert part_valids is not None
        assert n_valid is not None

        # keep only xyz coordinates if input has more channels
        gt_object_pcs = gt_object_pcs[:, :, :3]
        input_part_pcs = input_part_pcs[:, :, :3]

        # get fracture points info
        critical_points_idx = data_dict.get('critical_pcs_idx', None).cpu().numpy()
        n_critical_points = data_dict.get('n_critical_pcs', None).cpu().numpy()

        predicted = estimate_global_transform(
            pred_pairwise_matches,
            input_part_pcs,
            n_valid,
            n_pcs,
            critical_points_idx,
            n_critical_points,
            gt_part_rot,
            gt_part_trans,
        )

        return predicted

    def loss_function(self, data_dict, optimizer_idx, mode):
        """
        Orchestrates forward pass and loss computation.

        Input:
            data_dict: input data dictionary
            optimizer_idx: index of the optimizer (if multiple optimizers are used)
            mode: 'train', 'val', or 'test'

        Output:
            loss_dict: dictionary with losses and metrics
        """
        # forward pass with segmentation & matching prediction
        out_dict = self.forward(data_dict)

        # compute loss
        loss_dict = self._loss_function(data_dict, out_dict, optimizer_idx)

        # add batch size to loss_dict for weighted averaging
        if not self.training:
            if 'batch_size' not in loss_dict:
                loss_dict['batch_size'] = out_dict['batch_size']

        # during testing, perform global alignment and compute metrics
        if mode == 'test':
            pred_transforms = self.global_alignment(data_dict, out_dict)
            metrics_dict = self.calc_metric(data_dict, pred_transforms)
            loss_dict.update(metrics_dict)

        return loss_dict

    def forward_pass(self, data_dict, optimizer_idx, mode):
        """
        Wrapper for forward pass and loss computation.

        Input:
            data_dict: input data dictionary
            optimizer_idx: index of the optimizer (if multiple optimizers are used)
            mode: 'train', 'val', or 'test'

        Output:
            loss_dict: dictionary with losses and metrics
        """

        loss_dict = self.loss_function(data_dict, optimizer_idx, mode)
        
        if mode == "train" and self.local_rank == 0:
            # log losses for monitoring
            log_dict = {
                f'{mode}/{key}': value.item() if isinstance(value, torch.Tensor) else value
                for key, value in loss_dict.items()
            }

            # log data loading time
            time_data = [
                key for key in self.trainer.profiler.recorded_durations.keys() if 'prepare_data' in key
            ][0]
            log_dict[f'{mode}/data_time'] = self.trainer.profiler.recorded_durations[time_data][-1]

            self.log_dict(log_dict, logger=True, sync_dist=False, rank_zero_only=True)

        return loss_dict

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        Optimizer: Adam with lr=1e-4
        Weight_decay:
        - applied selectively (not to batch norm) on trainable parameters like weights
        - helps regularization and prevents overfitting
        LR Scheduler: 
        - cosine annealing with warm-up
        - warm-up for first 5% of epochs: linear increase from 0 to max_lr
        - cosine decay from max_lr to min_lr over remaining epochs: it keeps 
        lr high initially, avoids getting stuck in bad local minima, and then 
        decays faster in later stages; thus it's better than linear decay
        - helps stabilize training of the multi-task loss

        Output:
            optimizer: Adam/AdamW optimizer
            scheduler: cosine annealing scheduler
        """
        learning_rate = self.cfg.TRAIN.LR # 1e-4
        weight_decay = self.cfg.TRAIN.WEIGHT_DECAY # 1e-4

        # separate parameters for weight decay: apply weight decay only to weights, not biases or batch norm params
        if weight_decay > 0:
            params = filter_weight_decay_params(self)
            wd_params = [
                {
                    'params': params['decay'], 
                    'weight_decay': weight_decay
                },
                {
                    'params': params['no_decay'], 
                    'weight_decay': 0.0
                }
            ]
            optimizer = optim.AdamW(wd_params, lr=learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.0)

        # learning rate scheduler: cosine annealing with warm-up
        if self.cfg.TRAIN.LR_SCHEDULER:
            assert self.cfg.TRAIN.LR_SCHEDULER.lower() in ['cosine']

            total_epochs = self.cfg.TRAIN.NUM_EPOCHS # defaut 250
            warmup_epochs = int(total_epochs * self.cfg.TRAIN.WARMUP_RATIO) # 5% warm-up

            # cosine annealing with warm-up restarts
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                total_epochs,
                max_lr=learning_rate,
                min_lr=learning_rate / self.cfg.TRAIN.LR_DECAY_RATE,
                warmup_steps=warmup_epochs,
            )

            return (
                [optimizer],
                [
                    {
                        'scheduler': scheduler,
                        'interval': 'epoch'
                    }
                ]
            )

        return optimizer