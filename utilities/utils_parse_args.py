import os
import argparse
import platform
from .utils_config import CONFIG, config_from_file, config_from_list


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--cfg',
        '--config',
        dest='cfg_file',
        action='append',
        help='an optional config file',
        default=None,
        type=str
    )
    args = parser.parse_args()

    # load config from file
    if args.cfg_file is not None:
        for f in args.cfg_file:
            config_from_file(f)

    # generate output paths if model name is specified
    if len(CONFIG.MODEL_NAME) != 0:
        model_save_path = os.path.join('results', CONFIG.MODEL_NAME, 'model_save')
        results_save_path = os.path.join('results', CONFIG.MODEL_NAME)
        
        config_from_list(['OUTPUT_PATH', results_save_path, 'MODEL_SAVE_PATH', model_save_path])

        if not os.path.exists(CONFIG.MODEL_SAVE_PATH):
            os.makedirs(CONFIG.MODEL_SAVE_PATH, exist_ok=True)
        if not os.path.exists(CONFIG.OUTPUT_PATH):
            os.makedirs(CONFIG.OUTPUT_PATH, exist_ok=True)

    # copy the config file into the output path
    for file in args.cfg_file:
        if platform.system().lower() == "windows":
            cmd = f"copy {file} {CONFIG.OUTPUT_PATH}"
        else:
            cmd = f"cp {file} {CONFIG.OUTPUT_PATH}"
        os.system(cmd)

    return args