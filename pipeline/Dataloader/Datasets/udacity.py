from typing import Tuple, Callable

import math

import numpy as np
import random
import os

from bisect import bisect_left
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from pipeline.Dataloader.abstract_dataloader import AbstractDataloader


def take_closest(myList, myNumber):
    """
    TODO: refactor
    Code from: https://stackoverflow.com/a/12141511
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def get_closest_labels(image_list, y_map):
    """
    The labels were collected asynchrously from the camera feed, so the timestamps do not exactly align.
    To generate the label list, we take the closest label by timestamp for each image.
    """
    return [y_map[take_closest(list(y_map.keys()), int(image.stem))] for image in image_list]


class Udacity(AbstractDataloader):
    def __init__(self, input_path, sgs_path=None, label_path=None, batch_size=32, shuffle=False, set_splits=None, loader_type=None, bounds=None, max_steering=None, normalize_labels=False, *args, **kwargs) -> None:
        self.num = 0
        self.batch_size = batch_size
        self.img_list = []
        self.label_list = []
        self.sg_list = []
        self.input_path = input_path
        self.sgs_path = sgs_path
        self.label_path = label_path
        self.loader_type = "Default"
        self.strategy_types = {
            "Default":self._strategy_default,
            "Paths":self._strategy_paths,
            "Training":self._strategy_training
        }

        # Check that set_splits contain valid split names
        if set_splits:
            for split in set_splits:
                assert split in ["train", "test"], f"Udacity does not have a {split} split."

        # Load images
        if input_path.is_dir():
            for split in sorted(os.listdir(input_path)):
                if set_splits and split not in set_splits:
                    continue
                split_folder = input_path/split
                if (split_folder).is_dir():
                    full_split_folder = split_folder/'center'
                    for img_name in sorted(os.listdir(full_split_folder)):
                        if img_name.endswith(".jpg"):
                            img_path = full_split_folder/img_name
                            self.img_list.append(img_path)
        elif input_path.is_file():
            self.img_list.append(input_path)
        assert len(self.img_list) > 0, 'Found no images, exiting'
        # Load scene graphs if sgs_path is not None
        if sgs_path:
            if sgs_path.is_dir():
                for split in sorted(os.listdir(sgs_path)):
                    split_folder = sgs_path/split
                    if (split_folder).is_dir():
                        full_split_folder = split_folder/'center'
                        for sg_name in sorted(os.listdir(full_split_folder)):
                            if sg_name.endswith(".pkl"):
                                sg_path = full_split_folder/sg_name
                                self.sg_list.append(sg_path)
            elif sgs_path.is_file():
                self.sg_list.append(sgs_path)
            assert len(self.sg_list) == len(self.img_list), f"Found {len(self.img_list)} images, but {len(self.sg_list)} sgs, exiting"
        # Load labels if label_path is not None
        if label_path:
            if label_path.is_file() and label_path.suffix == ".csv":
                with open(label_path) as f:
                    steering_data = {}
                    for line in f.readlines()[1:]:
                        splits = line.split(',')
                        frame_id = int(splits[0])
                        steering_angle = splits[1]
                        steering_data[frame_id] = float(steering_angle)
                # Sort the keys to make sure they are in sorted order for later
                frame_ids = list(steering_data.keys())
                sorted_frame_ids = sorted(frame_ids)
                steering_data = {key: steering_data[key] for key in sorted_frame_ids}
                labels = get_closest_labels(self.img_list, steering_data)
                self.label_list = labels
            assert len(self.img_list) == len(self.label_list), f"Found {len(self.img_list)} images, but {len(self.label_list)} labels, exiting."
            if max_steering is not None:
                max_steering_rad = np.deg2rad(max_steering)
                indices_to_keep = [i for i in range(len(self.label_list))
                                   if -max_steering_rad <= self.label_list[i] <= max_steering_rad]
                print('%d images removed for having steering angle greater than %0.2f' %
                             (len(self.label_list) - len(indices_to_keep), max_steering))
                self.img_list = [self.img_list[i] for i in indices_to_keep]
                if normalize_labels:
                    self.label_list = [self.label_list[i] / max_steering_rad for i in indices_to_keep]
                else:
                    self.label_list = [self.label_list[i] for i in indices_to_keep]
            elif normalize_labels:
                raise ValueError('Can only normalize labels if max_steering is provided')

        if bounds is not None:
            self.set_bounds(bounds)

        if loader_type is not None:
            self.set_strategy(loader_type)

        if shuffle:
            random.shuffle(self.img_list)

    def __iter__(self):
        return super().__iter__()

    def __next__(self):
        if self.num < len(self.img_list):
            num = self.num
            self.num += self.batch_size
            batch = self._load(num, num+self.batch_size)
            return batch
        else:
            self.num = 0
            raise StopIteration

    def __len__(self):
        return len(self.img_list)

    def set_bounds(self, bounds):
        # 1 Tuple -> get slice withing lb : ub
        if isinstance(bounds, tuple):
            lb, ub = bounds
            self.img_list = self.img_list[lb:ub]
            if len(self.sg_list) > 0:
                self.sg_list = self.sg_list[lb:ub]
            if len(self.label_list) > 0:
                self.label_list = self.label_list[lb:ub]
        # 2 Tuples -> get slice lb0 : ub0 Union lb1 : ub1
        elif isinstance(bounds, list) and len(bounds) == 2:
            lb0, ub0 = bounds[0]
            lb1, ub1 = bounds[1]
            self.img_list = self.img_list[lb0:ub0] + self.img_list[lb1:ub1]
            if len(self.sg_list) > 0:
                self.sg_list = self.sg_list[lb0:ub0] + self.sg_list[lb1:ub1]
            if len(self.label_list) > 0:
                self.label_list = self.label_list[lb0:ub0] + self.label_list[lb1:ub1]

    def _load(self, lb, ub) -> Tuple[list,list]:
        strategy = self._load_strategy()
        return strategy(lb, ub)

    def _load_strategy(self) -> Callable[[int,int], Tuple[list,list]]:
        return self.strategy_types[self.loader_type]

    def _strategy_default(self, lb, ub):
        _img_list = []
        _img_name_list = []
        for img_path in self.img_list[lb:ub]:
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            _img_list.append(img)
            img_name = str(img_path.relative_to(self.input_path))[:-4]
            _img_name_list.append(img_name)
        return (_img_list, _img_name_list)

    def _strategy_paths(self, lb, ub):
        return (self.img_list[lb:ub], self.sg_list[lb:ub])

    def _strategy_training(self, lb, ub):
            _img_list = []
            _label_list = []
            for img_path, label in zip(self.img_list[lb:ub], self.label_list[lb:ub]):
                img = Image.open(img_path).convert('RGB')
                # Resize image to match Rambo sizing for Udacity
                img = img.resize((256,192))
                # Normalize images for training
                img = np.array(img) / 255.0
                # Move RGB channel to the beginning. Needed for training models.
                img = np.moveaxis(img, 2, 0)
                _img_list.append(img)
                _label_list.append(label)
            return (_img_list, _label_list)

    def set_strategy(self, s_type) -> None:
        if s_type not in self.strategy_types.keys():
            raise ValueError(f"{s_type} is not a valid loader type. Use 'Default', 'Paths' or 'Training'.")
        else:
            if s_type == "Paths" and self.sgs_path is None:
                raise AssertionError("SGs base path missing!")
            if s_type == "Training" and self.label_path is None:
                raise AssertionError("Labels path missing!")
            else:
                self.loader_type = s_type

    def get_number_batches(self):
        return math.ceil(len(self.img_list) / self.batch_size)
