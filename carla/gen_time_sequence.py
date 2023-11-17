import argparse
import os
import sys
from collections import defaultdict

from tqdm import tqdm

from utils.dataset import Dataset
from pathlib import Path, PosixPath

def custom_argparse(arg_string):
    parser = argparse.ArgumentParser(
        description="Time series from cluster"
    )
    parser.add_argument(
        "-dataset",
        "--dataset",
        type=Path,
        required=False,
        help="Cluster file"
    )
    parser.add_argument(
        "-time",
        "--time",
        type=int,
        required=False,
        help="Number of time steps to consider"
    )
    parser.add_argument(
        "-output",
        "--output_path",
        type=Path,
        required=True,
        help="Directory to save"
    )
    return parser.parse_args(arg_string)

def image_to_int(dataset: Dataset):
    image_map = dataset.image_to_cluster_map()
    map_values = list(set(image_map.values()))
    value_to_int = {val: index for index, val in enumerate(map_values)}
    int_map = {}
    for k, v in tqdm(image_map.items()):
        int_map[k] = value_to_int[v]
    return int_map

def parse_data(arg_string):
    args = custom_argparse(arg_string)
    output_path = args.output_path/''
    os.makedirs(output_path, exist_ok=True)
    orig_dataset = Dataset.load_from_file(args.dataset)
    orig_int_map = image_to_int(orig_dataset)

    tuple_map = defaultdict(list)
    for image in tqdm(orig_int_map):
        image_str = str(image)
        under_index = image_str.rfind('_')
        frame = int(image_str[under_index + 1:-4])
        key = []
        for back_time in range(args.time - 1):
            prev_image = PosixPath(f'{image_str[:under_index]}_{frame-(back_time-1)*4}.png')
            prev_cluster = orig_int_map[prev_image] if prev_image in orig_int_map else -1
            # print(back_time, frame, frame-back_time-1, prev_image, prev_cluster)
            key.insert(0, prev_cluster)
        key.append(orig_int_map[image])
        key = tuple(key)
        tuple_map[key].append(image)

    dataset = Dataset()
    dataset._image_files = list(orig_int_map.keys())
    dataset._sg_files = {img: img for img in dataset._image_files}
    dataset._has_clusters = True

    for value, list_of_images in tuple_map.items():
        list_of_images = sorted(list_of_images)
        dataset._clusters[list_of_images[0]] = list_of_images
        dataset._graph_metadata_groups[value] = list_of_images
    dataset._sorted_cluster_keys = sorted(dataset._clusters.keys(),
                                          key=lambda img_file: (len(dataset._clusters[img_file]), img_file),
                                          reverse=True)
    file_name = f'{args.dataset.stem}_time_{args.time}.json'
    dataset.save_to_file(str((output_path/file_name).absolute()), '', '')
    print(file_name, len(dataset), 'Original:', len(orig_dataset._clusters), 'New:', len(dataset._clusters))





if __name__ == '__main__':
    parse_data(sys.argv[1:])