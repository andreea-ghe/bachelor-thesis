from .joint_segmentation_align_model import JointSegmentationAlignModel


def build_jigsaw_model(config):
    model = JointSegmentationAlignModel(config)
    return model