import argparse
import copy
import os
import sys
from collections import defaultdict

from utils.dataset import Dataset
from pathlib import Path
import json

def custom_argparse(arg_string):
    parser = argparse.ArgumentParser(
        description="PhysCov data parser"
    )
    parser.add_argument(
        "-input",
        "--input_path",
        type=Path,
        required=False,
        help="Location of input images"
    )
    parser.add_argument(
        "-output",
        "--output_path",
        type=Path,
        required=True,
        help="Directory to save figures"
    )
    return parser.parse_args(arg_string)


def parse_data(arg_string):
    args = custom_argparse(arg_string)
    output_path = args.output_path/''
    for json_file in ['rss_dict', 'rss_v2_dict']:
        input_path = args.input_path/f'{json_file}.json'
        os.makedirs(output_path, exist_ok=True)
        with open(str(input_path.absolute()), 'r') as input_file:
            rss_data = json.load(input_file)
        for beam_count in range(1, 11):
            data = rss_data[f'{beam_count}']
            for key, value in data.items():
                data[key] = tuple([int(v) for v in value])
            dataset = Dataset()
            dataset._image_files = list(data.keys())
            dataset._sg_files = {img: img for img in dataset._image_files}
            dataset._has_clusters = True
            data_map = defaultdict(list)
            for key, value in data.items():
                data_map[value].append(key)

            for value, list_of_images in data_map.items():
                list_of_images = sorted(list_of_images)
                dataset._clusters[list_of_images[0]] = list_of_images
                dataset._graph_metadata_groups[value] = list_of_images
            dataset._sorted_cluster_keys = sorted(dataset._clusters.keys(),
                                                  key=lambda img_file: (len(dataset._clusters[img_file]), img_file),
                                                  reverse=True)
            dataset.save_to_file(str((output_path/f'{json_file[:-5]}_{beam_count}.json').absolute()), '', '')
            print(json_file, beam_count, len(dataset), len(dataset._clusters))





if __name__ == '__main__':
    parse_data(sys.argv[1:])