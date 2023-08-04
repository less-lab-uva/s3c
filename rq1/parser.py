import argparse
import ast

from pathlib import Path

DROP_STRATEGIES = [
    "random_drop",
    "front",
    "tail",
    "even_front",
    "even_tail",
    "front_label",
    "tail_label",
    "even_front_label",
    "even_tail_label",
    "front_both",
    "tail_both",
    "even_front_both",
    "even_tail_both",
    "binary",
    "binary_label",
    "binary_both"
]

def _percentFloat(f):
    f = float(f)
    if f < 0.0 or f > 1.0:
        raise argparse.ArgumentTypeError('Value has to be between 0 and 1')
    return f


def _bounds(string):
    lb, ub = string.split('(')[1].split(')')[0].split(',')
    lb = int(lb)
    ub = int(ub)
    return lb, ub


def _array(string):
    try:
        l = ast.literal_eval(string)
    except:
        l = string.split(',')
    return l


def custom_argparse(arg_string):
    parser = argparse.ArgumentParser(
        description="RQ1 Training pipeline"
    )
    parser.add_argument(
        "-dt",
        "--dataset_type",
        type=str,
        choices=['Udacity', 'Sully', 'CommaAi', 'CarlaRSV'],
        required=True,
        help="Dataset type"
    )
    parser.add_argument(
        "-dip",
        "--dataset_image_path",
        type=Path,
        required=True,
        help="Dataset image folder path"
    )
    parser.add_argument(
        "-dlp",
        "--dataset_label_path",
        type=Path,
        required=True,
        help="Dataset label folder path"
    )
    parser.add_argument(
        "-sgp",
        "--scene_graph_path",
        type=Path,
        required=True,
        help="Scene graph folder path"
    )
    parser.add_argument(
        "-cp",
        "--cluster_path",
        type=Path,
        required=True,
        help="Cluster .json path"
    )
    parser.add_argument(
        "-ds",
        "--drop_strategy",
        type=str,
        choices=DROP_STRATEGIES,
        required=False,
        default=None,
        help="Drop strategy"
    )
    parser.add_argument(
        '-ms',
        '--max_steering',
        type=float,
        required=False,
        default=25.0,
        help='Maximum steering angle in degrees. Images with a steering angle outside +/- max_steering are pruned.'
             'Default is 25.0, the Udacity default.'
    )
    parser.add_argument(
        '-norm',
        '--normalize_labels',
        type=bool,
        required=False,
        default=True,
        help='Whether to normalize the labels so that they fall within -1 and 1. Must provide max_steering if True.'
             'Default is True, will normalize to the default max_steering of 25.'
    )
    parser.add_argument(
        "-da",
        "--drop_amount",
        type=_percentFloat,
        required=False,
        help="Percentage of data dropped. Percentage should be any value between 0 and 1. Example: 0.5 drops 50% of the data"
    )
    parser.add_argument(
        "-sn",
        "--split_number",
        type=int,
        required=True,
        help="Split number identifier."
    )
    parser.add_argument(
        "-vb",
        "--val_bounds",
        type=_bounds,
        required=True,
        help="Tuple of lower and upper bounds for Validation Dataloader. It is inclusive of the lower bound and exlusive of the upper bound. Example: (lb,ub)"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        required=True,
        help="Output folder path to save models and statistics."
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        required=False,
        default=128,
        help="Batch size for training or evaluating a model."
    )
    parser.add_argument(
        "-r",
        "--rerun",
        default=False,
        action="store_true",
        help="If supplied, re-run existing data. Otherwise, skip existing data."
    )
    ### Arguments for new_study.py
    # parser.add_argument(
    #     "--split_town",
    #     type=str,
    #     choices=["Town01", "Town02", "Town04", "Town10HD"],
    #     help="Select the Carla town you want to train your model with."
    # )
    parser.add_argument(
        "--split_town",
        type=_array,
        help="Select the Carla towns you want to train your model with."
    )
    parser.add_argument(
        "--town_complement",
        action='store_true',
        help='If supplied, use the complement of the split_town for training, and leave split_town for test.'
        )
    parser.add_argument(
        "--split_percent",
        type=float,
        help="Select the percentage of data you want for a train split, the rest will be your test split."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of workers for the torch dataloader."
    )

    return parser.parse_args(arg_string)