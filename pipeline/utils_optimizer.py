import torch
from torch.nn import LayerNorm, GroupNorm, _BatchNorm, _InstanceNorm


def tensor_in_list(tensor: torch.Tensor, lst: list):
    """Judge whether a tensor is in a list."""
    for elem in lst:
        if tensor.shape == elem.shape and (tensor == elem).all():
            return True
    return False


def filter_weight_decay_params(model):
    """
    Separate model parameters into two groups: those that will have weight decay applied,
    and those that will not (biases and LayerNorm/BatchNorm weights).
    """
    # we need to sort the names so that we can save/load ckpts properly
    norm_module_names = []
    bias_module_names = []
    for name, module in model.named_modules():
        # iterate over modules, not parameters, to identify normalization layers, 
        # modules that contain bias paramters

        if isinstance(module, (LayerNorm, GroupNorm, _BatchNorm, _InstanceNorm)):
            # exclude norm weights
            norm_module_names.append(name)

        if hasattr(module, 'bias') and module.bias is not None:
            # exclude bias parameters
            bias_module_names.append(f"{name}.bias")
    norm_module_names.sort()
    bias_module_names.sort()

    no_decay = [model.get_submodule(module).weight for module in norm_module_names] +
                [model.get_submodule(module).bias for module in bias_module_names]

    decay_name = []
    for name, param in model.named_parameters():
        # include all parameters that are NOT in no_decay
        if param.requires_grad and not tensor_in_list(param, no_decay):
            decay_name.append(name)
    decay_name.sort()

    decay = [model.get_parameter(name) for name in decay_name]


    return {'decay': list(decay), 'no_decay': list(no_decay)}