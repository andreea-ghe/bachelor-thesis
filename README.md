# Jigsaw: Learning to Assemble Multiple Fractured Objects

This repository contains a refactored implementation of the paper
"[Jigsaw: Learning to Assemble Multiple Fractured Objects](https://arxiv.org/abs/2305.17975)"
for the bachelor thesis project.

## Installation

This repository has been developed and tested with Python 3.8 and CUDA 11.3.
To set up the required environment, follow these steps:

1. Create a new Anaconda environment named `assembly`:
    ```shell
    conda env create -f environment.yaml
    conda activate assembly
    ```

## Datasets

We provide support for the [Breaking Bad Dataset](https://breaking-bad-dataset.github.io/).
For more information about data processing, please refer to the dataset website.
Please make sure you use the updated inner-face-free version, as our tests are all based on that version.

After processing the data, ensure that you have a folder named `data` with the following structure:
```
data
├── breaking_bad
│   ├── everyday
│   │   ├── BeerBottle
│   │   │   ├── ...
│   │   ├── ...
│   ├── artifact
│   │   ├── ...
│   ├── everyday.train.txt
│   ├── everyday.val.txt
│   ├── artifact.train.txt
│   ├── artifact.val.txt
│   └── ...
└── ...
```

The `everyday` subset is used for training. The `artifact` subset can be used for cross-domain evaluation.

## Pretrained Checkpoints

Pretrained model weights are provided in the `checkpoint/` directory:

| Model | Checkpoint File | Description |
|-------|-----------------|-------------|
| Jigsaw (Binary) | `checkpoint/jigsaw_4x4_128_512_250e_cosine_everyday.ckpt` | Binary segmentation model |
| Jigsaw (Multi) | `checkpoint/jigsaw_multi_4x4_128_512_250e_cosine_everyday.ckpt` | Multi-class segmentation model |

## Run the Experiments

### Evaluation

To evaluate a pretrained model, run:
```shell
python -m experiments.eval_model --cfg experiments/jigsaw_4x4_128_512_250e_cosine_everyday.yaml
```

Available configuration files:

| Config | Description |
|--------|-------------|
| `experiments/jigsaw_4x4_128_512_250e_cosine_everyday.yaml` | Jigsaw (binary) on everyday subset |
| `experiments/jigsaw_multi_4x4_128_512_250e_cosine_everyday.yaml` | Jigsaw (multi) on everyday subset |
| `experiments/jigsaw_4x4_128_512_250e_cosine_artifact.yaml` | Jigsaw (binary) on artifact subset |
| `experiments/jigsaw_multi_4x4_128_512_250e_cosine_artifact.yaml` | Jigsaw (multi) on artifact subset |

The evaluation results will be stored in the `results/MODEL_NAME/` directory.

### Training

To train a model from scratch, run:
```shell
python -m experiments.train_model --cfg experiments/jigsaw_4x4_128_512_250e_cosine_everyday.yaml
```

Replace the config path with your desired configuration file.

## Code Structure

```
bachelor-thesis
├── base_pipeline/
│   ├── base_model.py           # Base model class
│   ├── utils_evaluation.py     # Evaluation utilities
│   ├── utils_lr_scheduler.py   # Learning rate schedulers
│   ├── utils_optimizer.py      # Optimizer utilities
│   ├── utils_transform.py      # Transform utilities
│   └── utils.py                # General utilities
├── checkpoint/
│   ├── jigsaw_4x4_128_512_250e_cosine_everyday.ckpt
│   └── jigsaw_multi_4x4_128_512_250e_cosine_everyday.ckpt
├── dataset_preprocessing/
│   ├── dataset_config.py       # Dataset configuration
│   └── fracture_assembly_dataset.py  # Dataset implementation
├── experiments/
│   ├── eval_model.py           # Evaluation script
│   ├── train_model.py          # Training script
│   ├── model_config.py         # Model configuration
│   └── *.yaml                  # Experiment configurations
├── feature_extractor/
│   └── ...                     # Feature extraction modules
├── global_alignment/
│   ├── estimate_global_poses.py
│   ├── spanning_tree_alignment.py
│   ├── utils_alignment.py
│   ├── utils_pose_graph.py
│   └── utils_shonan.py         # Shonan rotation averaging
├── jigsaw_pipeline/
│   ├── joint_segmentation_align_model.py  # Main Jigsaw model
│   ├── utils_losses.py         # Loss functions
│   ├── utils_pairwise_alignment.py
│   └── utils.py
├── multipart_matching/
│   ├── affinity.py             # Affinity computation
│   ├── utils_hungarian.py      # Hungarian algorithm
│   └── utils_sinkhorn.py       # Sinkhorn algorithm
├── surface_segmentation/
│   ├── segmentation_classifier.py
│   └── utils.py
├── utilities/
│   ├── rotation.py             # Rotation utilities
│   ├── utils_config.py         # Config utilities
│   ├── utils_edict.py          # EasyDict utilities
│   ├── utils_parse_args.py     # Argument parsing
│   └── utils_stdout.py         # Stdout utilities
├── results/                    # Output directory for results
├── environment.yaml            # Conda environment file
└── README.md                   # This file
```

## Configuration System

Configuration files are located in the `experiments/` directory.

Key configuration options:
```yaml
MODEL_NAME: your_model_name     # Folder name for results
MODULE: experiments             # Model module to load

DATASET: breaking_bad.all_piece_matching

GPUS: [0]                       # GPUs to use
BATCH_SIZE: 1
NUM_WORKERS: 0

TRAIN:
  NUM_EPOCHS: 250
  LR: 0.001
  LR_SCHEDULER: 'cosine'

DATA:
  SUBSET: everyday              # Dataset subset (everyday/artifact)
  MAX_NUM_PART: 20
  NUM_PC_POINTS: 5000
  SAMPLE_BY: area

MODEL:
  PC_CLS_METHOD: multi          # 'binary' or 'multi' for segmentation

WEIGHT_FILE: checkpoint/your_checkpoint.ckpt  # Path to pretrained weights
```

## Acknowledgement

This code is based on the [Jigsaw](https://github.com/Jiaxin-Lu/Jigsaw) repository.
