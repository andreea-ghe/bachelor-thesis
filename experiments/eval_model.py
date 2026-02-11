import os
import torch
import pytorch_lightning as pl
from datetime import datetime
from dataset_preprocessing import build_data_loaders, build_pairs_data_loaders
from utilities.utils_stdout import DuplicateStdoutFileManager
from utilities.utils_parse_args import parse_args
from utilities.utils_config import CONFIG
from utilities.utils_edict import print_edict
from jigsaw_pipeline import build_jigsaw_model
from pytorch_lightning.loggers import WandbLogger


NOW_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def test_model(config):
    """
    Evaluation function for the jigsaw model.
    Tests the trained model on the validation/test dataset and computes metrics:
    - Chamfer Distance (CD): overall alignment quality
    - Part Accuracy (PA): how many pieces matched correctly
    - Rotation Error (RE): accuracy of predicted rotations
    - Translation Error (TE): accuracy of predicted translations

    Notes:
    It uses the predicted fracture segmentation, not the ground truth.
    Applies Hungarian algorithm for discrete matching.
    Perform global alignment to recover all poses.

    Input:
        config: configuration object with test settings
    """
    if len(config.STATS):
        os.makedirs(config.STATS, exist_ok=True) # create stats directory if needed

    # Step 1: initialize data loaders
    # train_loader, val_loader = build_data_loaders(config)
    train_loader, val_loader = build_pairs_data_loaders(config)

    # Step 2: initialize model architecture
    # model will be populated with trained weights from checkpoint
    model = build_jigsaw_model(config)


    # setup logging
    model_save_path = config.MODEL_SAVE_PATH # for model checkpoints

    if config.LOG_FILE_NAME is not None and len(config.LOG_FILE_NAME) > 0:
        logger_name = f"{config.MODEL_NAME}_{config.LOG_FILE_NAME}"
    else:
        logger_name = f"{config.MODEL_NAME}_{NOW_TIME}"
    logger = WandbLogger(
        project=config.PROJECT,
        name=logger_name,
        id=None,
        save_dir = model_save_path,
    )

    callbacks = []

    # Step 3: setup PyTorch Lightning trainer for testing
    all_gpus = list(config.GPUS)
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=all_gpus,
        strategy="dp" if len(all_gpus) > 1 else None,
        callbacks=callbacks,
    )

    # Step 4: load trained model checkpoint
    # detects and loads the best trained model for evaluation
    ckp_files = os.listdir(model_save_path)
    ckp_files = [
        ckp for ckp in ckp_files if "model_" in ckp
    ]

    if config.WEIGHT_FILE: # load specified weight file
        ckp = torch.load(config.WEIGHT_FILE, map_location='cpu')

        if 'state_dict' in ckp: # full checkpoint provided
            ckp_path = config.WEIGHT_FILE
            # Debug: compare model and checkpoint keys
            ckp_keys = set(ckp['state_dict'].keys())
            model_keys = set(model.state_dict().keys())
            missing_in_ckp = model_keys - ckp_keys
            missing_in_model = ckp_keys - model_keys
            if missing_in_ckp:
                print(f"Keys in model but NOT in checkpoint ({len(missing_in_ckp)}):")
                for k in sorted(missing_in_ckp)[:10]:
                    print(f"  {k}")
            if missing_in_model:
                print(f"Keys in checkpoint but NOT in model ({len(missing_in_model)}):")
                for k in sorted(missing_in_model)[:10]:
                    print(f"  {k}")
            if not missing_in_ckp and not missing_in_model:
                print("All checkpoint keys match model keys!")
        else:
            ckp_path = None # only weights provided, not full checkpoint
            model.load_state_dict(ckp, strict=False)

    elif ckp_files: # auto detect latest checkponint
        ckp_files = sorted(
            ckp_files,
            key=lambda x: os.path.getmtime(os.path.join(model_save_path, x)),
        )
        last_ckp = ckp_files[-1]
        ckp_path = os.path.join(model_save_path, last_ckp)
    else: # no checkpoint found
        ckp_path = None

    # load model with trained weights
    model = model.load_from_checkpoint(checkpoint_path=ckp_path, strict=False, config=config)
    
    # STEP 5: Run evaluation
    # This will:
    # 1. Run forward pass with predicted segmentation
    # 2. Compute matching using Hungarian algorithm
    # 3. Perform global alignment (Section 3.4)
    # 4. Calculate all metrics (PA, RE, TE, CD)
    # Reports these metrics for comparison
    print("Finish Setting -----")
    trainer.test(model, val_loader)

    print("Done evaluation")


if __name__ == "__main__":
    """
    Main entry point for evaluating Jigsaw model.
    
    Usage:
        python eval_matching.py --cfg experiments/jigsaw_250e_cosine.yaml
        python eval_matching.py --cfg experiments/jigsaw_250e_cosine.yaml --weight path/to/checkpoint.ckpt
    
    The evaluation process:
    1. Load configuration and specify checkpoint
    2. Build model and load trained weights
    3. Run inference on test set
    4. Compute metrics: PA, RE, TE, CD
    5. Log results to WandB and file
    """
    # Parse command line arguments
    args = parse_args("Jigsaw")

    # Set random seed for reproducibility
    torch.manual_seed(CONFIG.RANDOM_SEED)

    # Setup evaluation log file
    file_end = NOW_TIME
    if CONFIG.LOG_FILE_NAME is not None and len(CONFIG.LOG_FILE_NAME) > 0:
        file_end += "_{}".format(CONFIG.LOG_FILE_NAME)
    full_log_name = f"eval_log_{file_end}"

    # Duplicate stdout to log file
    with DuplicateStdoutFileManager(os.path.join(CONFIG.OUTPUT_PATH, f"{full_log_name}.log")) as _:
        # Print configuration
        print_edict(CONFIG)  
        # Run evaluation
        test_model(CONFIG)
