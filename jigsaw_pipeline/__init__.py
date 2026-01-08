from .joint_segmentation_align_model import *
from .utils_losses import *
from .utils_pairwise_alignment import *
from .utils import *


def build_jigsaw_model(config):
    model = JointSegmentationAlignmentModel(config)
    return model