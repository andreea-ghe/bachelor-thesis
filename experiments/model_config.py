from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

__C.JIGSAW = edict()

# Rotation representation type
# Paper uses rotation matrices for final pose estimation
__C.JIGSAW.ROT_TYPE = 'rmat'

# Point-level feature dimension extracted by the front-end encoder
# Features are extracted at dimension d = 128
__C.JIGSAW.PC_FEAT_DIM = 128

# Affinity feature dimension after primal-dual descriptor
__C.JIGSAW.AFF_FEAT_DIM = 512

# Affinity computation method: 'aff_dual' uses the primal-dual descriptor
__C.JIGSAW.AFFINITY = 'aff_dual'

# Encoder backbone: PointNet++ with multi-scale grouping and dynamic point support
# Paper Section 3.1: "As illustrated in Figure 7 of the PointNet paper, 
# a global descriptor of each piece lives between the learned features and the second
# FC layer... we employ a multi-scale grouping PointNet++"
# Dynamic version handles variable number of points per piece
__C.JIGSAW.ENCODER = 'pointnet2_pt.msg.dynamic'

# Whether to use segmentation mask during testing
# If True, only fracture points are used for matching (more accurate)
__C.JIGSAW.TEST_S_MASK = True

# Point classification method for fracture surface segmentation
# 'binary': 2-class (fracture vs non-fracture)
# 'multi': multi-class segmentation (slight improvement ~1.0 in metrics)
__C.JIGSAW.PC_CLS_METHOD = 'binary'  # ['binary', 'multi']
__C.JIGSAW.PC_NUM_CLS = 2  # Number of classes for segmentation

# Sinkhorn algorithm parameters for soft matching
__C.JIGSAW.SINKHORN_MAXITER = 20  # Maximum iterations for Sinkhorn
__C.JIGSAW.SINKHORN_TAU = 0.05  # Temperature parameter τ

# Transformer attention layer parameters
# Multi-head attention with 8 heads
__C.JIGSAW.TF_NUM_HEADS = 8  # Number of attention heads
__C.JIGSAW.TF_NUM_SAMPLE = 16  # Number of neighbor samples for local feature aggregation

# Loss function weights and scheduling
# ℒ = αℒ_seg + βℒ_mat + γℒ_rig
__C.JIGSAW.LOSS = edict()

# Weight for segmentation loss (fracture surface classification)
__C.JIGSAW.LOSS.w_cls_loss = 1.0

# Weight for matching loss (multi-part matching)
# Starts at epoch 9 (warm-up period for segmentation first)
__C.JIGSAW.LOSS.w_mat_loss = 0.0  # Initial weight (increases after mat_epoch)
__C.JIGSAW.LOSS.mat_epoch = 9  # Epoch to start applying matching loss

# Weight for rigidity loss (geometric consistency)
# Starts at epoch 199 (applied near end of training for refinement)
__C.JIGSAW.LOSS.w_rig_loss = 0.0  # Initial weight (increases after rig_epoch)
__C.JIGSAW.LOSS.rig_epoch = 199  # Epoch to start applying rigidity loss

def get_model_config():
    """
    Get the Jigsaw model configuration.
    
    Returns:
        EasyDict containing all model hyperparameters
    """
    return model_cfg.JIGSAW
