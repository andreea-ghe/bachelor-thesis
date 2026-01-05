import os 
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from jigsaw_pipeline import build_jigsaw_model
from dataset_preprocessing import build_data_loaders
from datetime import datetime
from utilities import parse_args, print_edict, DuplicateStdoutFileManager


NOW_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def train_model(config):
    """
    Training function for the jigsaw model.
    Pipeline:
    1. Initialize data loaders with dataset.
    2. Build the jigsaw model.
    3. Configure PyTorch Lightning trainer with callbacks.
    4. Train with loss scheduling (seg -> match -> rigid).

    The training uses:
    - Adam optimizer with lr=1e-4
    - cosine annealing lr scheduler
    - gradient clipping
    - model checkpointing based on validation loss

    Input:
        config: configuration object with training parameters
    """
    # Step 1: Initialize data loaders
    # build dataloaders for Breaking Bad dataset with area based sampling
    train_loader, val_loader = build_data_loaders(config)

    # Step 2: Build the jigsaw model
    model = build_jigsaw_model(config)

    # configure output directories
    model_save_path = config.MODEL_SAVE_PATH # for model checkpoints
    results_save_path = config.OUTPUT_PATH # for logs and results

    if config.LOG_FILE_NAME is not None and len(config.LOG_FILE_NAME) > 0:
        logger_name = f"{config.MODEL_NAME}_{config.LOG_FILE_NAME}"
    else:
        logger_name = f"{config.MODEL_NAME}_{NOW_TIME}"
    logger = WandbLogger(
        project=config.PROJECT,
        name=logger_name,
        id=None,
        save_dir = results_save_path,
    )

    # Step 3: Configure PyTorch Lightning trainer with callbacks
    # model checkpointing: save best models during training
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="model{epoch:03d}",
        monitor=config.CALLBACK.CHECKPOINT_MONITOR,
        save_top_k=10, # keep top 10 best models
        mode=config.CALLBACK.CHECKPOINT_MODE,
        save_last=True, # always save last checkpoint
    )
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'), # track learning rate
        checkpoint_callback,
    ]

    # configure lightning trainer
    training_log_dict = {
        'logger': logger,
        'accelerator': 'gpu',
        'devices': list(config.GPUS),
        'max_epochs': config.TRAIN.NUM_EPOCHS,
        'callbacks': callbacks,
        'benchmark': config.CUDNN, # cudnn benchmark for speed
        'gradient_clip_val': config.TRAIN.CLIP_GRAD, # gradient clipping
        'check_val_every_n_epoch': config.TRAIN.VAL_EVERY, # validation frequency
        'log_every_n_steps': 10,
        'profiler': 'simple',
        'detect_anomaly': True,
    }

    trainer = pl.Trainer(**training_log_dict)

    ckp_files = os.listdir(model_save_path)
    ckp_files = [
        ckp for ckp in ckp_files if ("model_" in ckp) or ("last" in ckp)
    ]

    if config.WEIGHT_FILE: # load from specified checkpoint
        ckp = torch.load(config.WEIGHT_FILE, map_location='cpu')

        if 'state_dict' in ckp.keys():
            # full checkpoint with optimizer etc.
            ckp_path = config.WEIGHT_FILE
        else:
            # only model weights: start training from scratch with these weights
            ckp_path = None
            model.load_state_dict(ckp)
    elif ckp_files: # load from last checkpoint in model save path
        ckp_files = sorted(
            ckp_files,
            key=lambda x: os.path.getmtime(os.path.join(model_save_path, x))
        )
        last_ckp = ckp_files[-1]
        ckp_path = os.path.join(model_save_path, last_ckp)\
    else:
        ckp_path = None # start training from scratch

    # Step 4: Train the model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckp_path)
    print("Done training.")


if __name__ == "__main__":
    """
    Main entry point for training the jigsaw model.

    The training process:
    1. Load configuration from YAML file
    2. Set random seeds for reproducibility
    3. Adjust batch size for multi-GPU training
    4. Initialize logging
    5. Train model with scheduled losses
    """
    args = parse_args()

    # set random seeds for reproducibility
    pl.seed_everything(config.RANDOM_SEED)

    # setup logging file
    file_end = NOW_TIME
    if config.LOG_FILE_NAME is not None and len(config.LOG_FILE_NAME) > 0:
        file_end += "_{}".format(config.LOG_FILE_NAME)
    log_file = f"train_log_{file_end}"

    with DuplicateStdoutFileManager(os.path.join(config.OUTPUT_PATH, f"{log_file}.log")) as _:
        # print configuration
        print_easydict(config)

        train_model(config)