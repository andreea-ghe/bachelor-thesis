from .pointnet_architecture import PointNetPTMSG


def build_feature_extractor(arch, features_dimension, **kwargs):
    if 'in_feat_dim' in kwargs:
        input_features_dim = kwargs['in_feat_dim']
    else:
        input_features_dim = 3  # default

    if isinstance(features_dimension, list):
        model = PointNetPTMSG(features_dimension[0], features_dimension[1])
    else:
        model = PointNetPTMSG(input_features_dim, features_dimension)

    return model