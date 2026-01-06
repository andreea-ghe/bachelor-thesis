import yaml
import importlib
from easydict import EasyDict as edict
from ast import literal_eval


__C = edict()
CONFIG = __C # global config object

__C.MODEL_NAME = ""  # this name would be the result file name
__C.MODULE = ""  # sample: dgl.network  b_global.network

__C.BATCH_SIZE = 32
__C.NUM_WORKERS = 8

__C.LOG_FILE_NAME = ""  # the suffix of log file

__C.MODEL_SAVE_PATH = ""  # auto generated

#
# Dataset
#
__C.DATASET = ""

# Other dataset specific configs should be imported from dataset_config.py

# wandb project name
__C.PROJECT = ""

#
# Training options
#

__C.TRAIN = edict()

# Total epochs
__C.TRAIN.NUM_EPOCHS = 200

# Optimizer type
__C.TRAIN.OPTIMIZER = "SGD"

# Start learning rate
__C.TRAIN.LR = 0.001

# LR Scheduler
__C.TRAIN.LR_SCHEDULER = "cosine"

# Learning rate decay
__C.TRAIN.LR_DECAY = 100.0

# Learning rate decay step (in epochs)
__C.TRAIN.LR_STEP = [10, 20]

# warmup_ratio for Adam Cosine
__C.TRAIN.WARMUP_RATIO = 0.0

# clip_grad
__C.TRAIN.CLIP_GRAD = None

# beta1, beta2 for Adam Optimizer
__C.TRAIN.beta1 = 0
__C.TRAIN.beta2 = 0.9

# weight decay for Adam or SGD
__C.TRAIN.WEIGHT_DECAY = 0.0

# SGD momentum
__C.TRAIN.MOMENTUM = 0.9

# Check val every n epoch
__C.TRAIN.VAL_EVERY = 5

# Visualization during training
__C.TRAIN.VIS = True
__C.TRAIN.VAL_SAMPLE_VIS = 5

# Loss function.
__C.TRAIN.LOSS = ""

#
# Callback
#
__C.CALLBACK = edict()
__C.CALLBACK.MATCHING_TASK = ["trans"]
__C.CALLBACK.CHECKPOINT_MONITOR = "val/loss"
__C.CALLBACK.CHECKPOINT_MODE = "min"

#
# Loss config
#
__C.LOSS = edict()

#
# Evaluation options
#
__C.EVAL = edict()

#
# MISC
#
# Parallel GPU indices ([0] for single GPU)
__C.GPUS = [0]
# Parallel strategy for multiple gpus
__C.PARALLEL_STRATEGY = "ddp"

# Float Precision, 32 for False, 16 for True
__C.FP16 = False

# CUDNN benchmark
__C.CUDNN = False

__C.WEIGHT_FILE = ""

# Output path (for checkpoints, running logs)
__C.OUTPUT_PATH = ""

# The step of iteration to print running statistics.
# The real step value will be the least common multiple of this value and batch_size
__C.STATISTIC_STEP = 100

# random seed used for data loading
__C.RANDOM_SEED = 42

# directory for collecting statistics of results
__C.STATS = ""


def merge_configs(src, dest):
    """
    Merge source config into destination config recursively.
    """
    for key, value in src.items():
        if key not in dest:
            raise KeyError(f'Key {key} not a valid config key.')

        if type(dest[key]) is not type(value):
            if type(dest[key]) is float and type(value) is int:
                value = float(value)
            else:
                if key not in ['CLASS']:
                    raise ValueError(f'Type mismatch ({type(dest[key])} vs. {type(value)}) for config key: {key}')
        if type(value) is edict:
            try:
                merge_configs(src[key], dest[key])
            except:
                print(f'Error under config key: {key}')
                raise
        else:
            dest[key] = value


def config_from_file(filename):
    """
    Load configuration from a YAML file and merge it into the default config.
    """
    with open(filename, 'r') as f:
        config = edict(yaml.full_load(f))

    # dynamically import model and dataset config modules
    if 'MODULE' in config and 'MODEL' not in __C:
        model_config_module = '.'.join(['model'] + [config.MODULE.split('.')[0]] + ['model_config'])
        mod = importlib.import_module(model_config_module)
        __C['MODEL'] = mod.get_model_config()

    if 'DATASET' in config and config.DATASET is not None:
        mod = importlib.import_module('dataset')
        __C['DATA'] = mod.dataset_config[config.DATASET.split('.')[0].upper()]

    merge_configs(config, __C)

def config_from_list(config_list):
    """
    It takes a flat list of strings, interprets them as key-value pairs, and overwrites 
    values inside a nested configuration dictionary in a safe manner.
    """
    assert len(config_list) % 2 == 0 # even number of arguments 
    
    for key, v in zip(config_list[0::2], config_list[1::2]): # even indices are keys, odd indices are values
        key_list = key.split('.') # split by dot to get nested keys
        d = __C

        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]

        subkey = key_list[-1]
        assert subkey in d.keys()

        try:
            value = literal_eval(v)
        except:
            value = v

        assert type(value) == type(d[subkey]), f'Type mismatch ({type(d[subkey])} vs. {type(value)}) for config key: {key}'
        d[subkey] = value