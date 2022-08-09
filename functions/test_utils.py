from argparse import Namespace
import argparse


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(
        description="LMU-SS2022-CVDL-Project")

    parser.add_argument("--dataset_dir",
                        type=str,
                        default="datasets-processed",
                        help="Directory contains processed MS COCO dataset.")

    parser.add_argument("--save_dir",
                        type=str,
                        default="test_output",
                        help="Directory to save the output files.")

    parser.add_argument("--config_path",
                        type=str,
                        default="config.json",
                        help="Path for the configuration json file.")

    parser.add_argument("--checkpoint_name",
                        type=str,
                        default="0108.0132/checkpoint_best.pth.tar",
                        help="Path for the checkpoint file.")

    parser.add_argument(
        '--device',
        type=str,
        default="gpu",  # gpu, cpu
        help='parameter of device')

    args = parser.parse_args()

    return args
