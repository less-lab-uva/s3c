import logging
import os
import sys

import argparse
from pathlib import Path

from pipeline.Dataloader.dataloader_factory import DataLoaderFactory, DATALOADERS
from utils.dataset import Dataset


def parse_cluster_args(arg_string):
    parser = argparse.ArgumentParser(description="Arguments for cluster generation")
    parser.add_argument(
        "-dt",
        "--dataset_type",
        type=str,
        choices=[dataloader for dataloader in DATALOADERS],
        required=False,
        help="Dataset type"
    )
    parser.add_argument(
        "-dp",
        "--dataset_path",
        type=Path,
        required=False,
        help="Dataset folder path"
    )
    parser.add_argument(
        "-sp",
        "--sgs_path",
        type=Path,
        required=False,
        help="Path to save/load SGs"
    )
    parser.add_argument(
        "-dsf",
        "--dataset_file",
        type=Path,
        required=False,
        help="Location to store data set file. If file exists, process will not regenerate and will exit. If file does "
             "not exist, then dataset_type, dataset_path, and sgs_path must be set."
    )
    parser.add_argument(
        "-j",
        "--threads",
        type=int,
        required=False,
        default=1,
        help="Number of threads to use for various operations."
    )
    parser.add_argument(
        "-max_per_thread",
        "--max_per_thread",
        type=int,
        required=False,
        default=512,
        help="Maximum number of graphs per thread. Larger groups will be split across threads and then merged."
    )
    parser.add_argument(
        "-load_threaded",
        "--load_threaded",
        type=bool,
        required=False,
        default=False,
        help="Whether to load the SGs in a threaded manner."
    )
    parser.add_argument('-v', '--verbose', action='store_true')

    return parser.parse_args(arg_string)


def generate_clusters(arg_string):
    args = parse_cluster_args(arg_string)
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    if args.dataset_file is not None and os.path.exists(args.dataset_file):
        # The below structure can be used to load the datasets created by this method.
        # Note that the dataset path and sgs path args only need to be provided if the full file paths are needed
        # Otherwise, the dataset contains sufficient information for most other analysis
        dataset = Dataset.load_from_file(args.dataset_file, args.dataset_path, args.sgs_path)
    else:
        logging.info('Could not find dataset file %s, generating dataset' % args.dataset_file)
        dataloader = DataLoaderFactory(args.dataset_type, args.dataset_path, sgs_path=args.sgs_path,
                                       loader_type='Paths', shuffle=False)
        dataset = Dataset(dataloader,
                          threads=args.threads,
                          max_per_thead=args.max_per_thread,
                          load_threaded=args.load_threaded)
        if args.dataset_file is not None:
            logging.info('Saving dataset to file')
            dataset.save_to_file(args.dataset_file, args.dataset_path, args.sgs_path)


if __name__ == '__main__':
    generate_clusters(sys.argv[1:])
