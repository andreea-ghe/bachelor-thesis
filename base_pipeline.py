import os
import pickle
import torch
import pytorch_lightning
from utilities.rotation import Rotation3D


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
        - Chamfer Distance (CD): Point cloud alignment quality  
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
        metric_dict = dict()

        # convert predictions to tensors
        predicted_trans = torch.tensor(trans_dict['trans'], dtype=torch.float32, device=part_pcs.device)
        predicted_rot = torch.tensor(trans_dict['rot'], dtype=torch.float32, device=part_pcs.device)
        predicted_rot = Rotation3D(predicted_rot, rot_type='rmat')

        # ground truth values
        gt_trans = data_dict['part_trans']
        gt_rot = data_dict['part_rot']

        N_SUM = part_pcs.shape[1] # number of pieces per object
        n_pcs = data_dict['n_pcs'] # [B] number of valid pieces per object
        B, P = n_pcs.shape

        # resample point clouds to have same nr of points per piece; easier CD computation
        nr_points_pcs_resampled = []
        for b in range(B):
            point_sum = 0
            resampled_pcs = []
            for p in range(P):
                