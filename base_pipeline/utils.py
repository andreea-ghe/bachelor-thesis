import numpy as np
import torch


def dict_to_numpy(data_dict):
    """
    """
    numpy_dict = {}
    
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            numpy_dict[key] = value.detach().cpu().numpy()
        elif isinstance(value, str):
            numpy_dict[key] = value
        elif isinstance(value, np.ndarray):
            numpy_dict[key] = value
        elif isinstance(value, list):
            if isinstance(value[0], torch.Tensor):
                value_device = [t.detach().cpu().numpy() for t in value]
                numpy_dict[key] = value_device
            else:
                numpy_dict[key] = value
        else:
            numpy_dict[key] = value
    
    return numpy_dict