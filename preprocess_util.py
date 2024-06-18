import argparse
import os
import yaml

def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):

    if not args.config_file:
        parser.error("Missing required argument --config_file")
        # return args
    
    config_path = args.config_file + ".yaml" if not args.config_file.endswith(".yaml") else args.config_file

    if not os.path.exists(config_path):
        print(f"{config_path} not found.")
        exit(1)

    print(f"Loading settings from {config_path}...")
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    ignore_nesting_dict = {}
    for section_name, section_dict in config_dict.items():
        # if value is not dict, save key and value as is
        if not isinstance(section_dict, dict):
            ignore_nesting_dict[section_name] = section_dict
            continue

        # if value is dict, save all key and value into one dict
        for key, value in section_dict.items():
            ignore_nesting_dict[key] = value

    config_args = argparse.Namespace(**ignore_nesting_dict)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]
    print(args.config_file)

    return args

def add_preprocess_arguments(parser: argparse.ArgumentParser):
    # parser.add_argument(
    #     "--config_file",
    #     type=str,
    #     default=None,
    #     help="using .yaml instead of args to pass hyperparameter",
    # )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="using .yaml instead of args to pass hyperparameter",
    )
    parser.add_argument(
        '-i', '--input',
        type=str, 
        default=None, 
        help='input image/folder path.')
    parser.add_argument(
        '-r', '--ref', 
        type=str, 
        default=None, 
        help='reference image/folder path if needed.')
    parser.add_argument(
        '--metric_mode',
        type=str,
        default='FR',
        help='metric mode Full Reference or No Reference. options: FR|NR.')
    parser.add_argument(
        '-m', '--metric_name', 
        type=str, 
        default='PSNR', 
        help='IQA metric name, case sensitive.')
    parser.add_argument(
        '--save_file', 
        type=str, 
        default=None, 
        help='path to save results.')