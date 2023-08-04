import argparse
import sys
from tqdm import tqdm

from utils.dataset import Dataset
from pathlib import Path
def custom_argparse(arg_string):
    parser = argparse.ArgumentParser(
        description="Validation Split Visualizer"
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        required=True,
        help="Location to load data set file."
    )
    return parser.parse_args(arg_string)


def compute_average(arg_string):
    args = custom_argparse(arg_string)
    dataset = Dataset.load_from_file(args.input_path, '', '')
    num_nodes = []
    num_edges = []
    reverse_map = {}
    for metadata, keys in tqdm(dataset._graph_metadata_groups.items()):
        reverse_map.update({key: metadata for key in keys})
    for index, key in enumerate(tqdm(dataset._sorted_cluster_keys)):
        nodes, edges, data = reverse_map[key]
        if index < 10:
            print(f'Index {index}: {data}')
        num_nodes.append(nodes)
        num_edges.append(edges)
    print(f'{args.input_path} has average {sum(num_nodes)/len(num_nodes):.2f} nodes and {sum(num_edges)/len(num_edges):.2f} edges per class.')


if __name__ == '__main__':
    compute_average(sys.argv[1:])