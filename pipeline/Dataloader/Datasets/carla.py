from typing import Tuple, Callable

import cv2
import math
import numpy as np
import pandas as pd
import random
import os
import torch

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pipeline.Dataloader.abstract_dataloader import AbstractDataloader
from torch.utils.data import Dataset, DataLoader

def get_key(filename):
  first = int(filename[:filename.find('_')])
  filename = filename[filename.find('_')+1:]
  filename = filename[filename.find('_')+1:]
  under_loc = filename.find('_')
  second = int(filename[:under_loc if under_loc > -1 else filename.find('.')])
  return (first, second, filename)

class Carla(AbstractDataloader):
    def __init__(self, input_path, sgs_path=None, label_path=None, batch_size=32, shuffle=False, set_splits=None, loader_type=None, bounds=None, max_steering=None, normalize_labels=False, type='rsv', save_paths_to_csv=False, *args, **kwargs) -> None:
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

        # Load images
        aux_c = 0
        if input_path.is_dir():
            if set_splits is None and (input_path / "img_paths.csv").is_file():
                print("Loading img paths from csv file")
                s = pd.read_csv(input_path / "img_paths.csv").squeeze()
                for p in s:
                    img_path = input_path / p
                    self.img_list.append(img_path)
            else:
                for town in sorted(os.listdir(input_path)):
                    if set_splits and town not in set_splits:
                        continue
                    town_folder = input_path/town
                    if town_folder.is_dir():
                        dir_list = os.listdir(town_folder)
                        sorted_dir_list = sorted(dir_list, key=get_key)
                        for img_name in sorted_dir_list:
                            if img_name.endswith(".png"):
                                img_path = town_folder/img_name
                                self.img_list.append(img_path)
                                aux_c += 1
                                print(f"Loaded img {aux_c}")
                # Save img paths to csv file
                if save_paths_to_csv:
                    img_path_list = []
                    for p in self.img_list:
                        p = str(p.relative_to(input_path))
                        img_path_list.append(p)
                    s = pd.Series(img_path_list)
                    s.to_csv(input_path / "img_paths.csv", index=False)
        elif input_path.is_file():
            self.img_list.append(input_path)
        assert len(self.img_list) > 0, 'Found no images, exiting'
        # Load scene graphs if sgs_path is not None
        if sgs_path:
            if sgs_path.is_dir():
                for town in sorted(os.listdir(sgs_path)):
                    if set_splits and town not in set_splits:
                        continue
                    town_folder = sgs_path/town
                    if town_folder.is_dir():
                        for sg_name in sorted(os.listdir(town_folder)):
                            if sg_name.endswith(f".{type}"):
                                sg_path = town_folder/sg_name
                                self.sg_list.append(sg_path)
            elif sgs_path.is_file():
                self.sg_list.append(sgs_path)
            assert len(self.sg_list) == len(self.img_list), f"Found {len(self.img_list)} images, but {len(self.sg_list)} sgs, exiting"
        # Load labels if label_path is not None
        aux_c = 0
        if label_path:
            if label_path.is_dir():
                if set_splits is None and (label_path / "labels.csv").exists():
                    print("Loading labels from csv file")
                    s = pd.read_csv(label_path / "labels.csv").squeeze()
                    self.label_list = s.values.tolist()
                else:
                    for town in sorted(os.listdir(label_path)):
                        if set_splits and town not in set_splits:
                            continue
                        town_folder = label_path/town
                        if town_folder.is_dir():
                            dir_list = os.listdir(town_folder)
                            sorted_dir_list = sorted(dir_list, key=get_key)
                            for label_name in sorted_dir_list:
                                if label_name.endswith(".steer"):
                                    label_file_path = town_folder/label_name
                                    with open(label_file_path) as f:
                                        label = float(f.readline())
                                        self.label_list.append(label)
                                        aux_c += 1
                                        print(f"Loaded label {aux_c}")
                    # Save img paths to csv file
                    if save_paths_to_csv:
                        s = pd.Series(self.label_list)
                        s.to_csv(label_path / "labels.csv", index=False)
            assert len(self.img_list) == len(self.label_list), f"Found {len(self.img_list)} images, but {len(self.label_list)} labels, exiting."
            # if max_steering is not None:
            #     max_steering_rad = np.deg2rad(max_steering)
            #     indices_to_keep = [i for i in range(len(self.label_list))
            #                        if -max_steering_rad <= self.label_list[i] <= max_steering_rad]
            #     print('%d images removed for having steering angle greater than %0.2f' %
            #                  (len(self.label_list) - len(indices_to_keep), max_steering))
            #     self.img_list = [self.img_list[i] for i in indices_to_keep]
            #     if normalize_labels:
            #         self.label_list = [self.label_list[i] / max_steering_rad for i in indices_to_keep]
            #     else:
            #         self.label_list = [self.label_list[i] for i in indices_to_keep]
            # elif normalize_labels:
            #     raise ValueError('Can only normalize labels if max_steering is provided')

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
            for img_path in self.img_list[lb:ub]:
                # img = Image.open(img_path).convert('RGB')
                img = cv2.imread(str(img_path))
                # Crop lower part of the image to remove the car hood/roof
                img = img[:418,:,:]
                # Resize image to match Rambo sizing for Udacity
                # img = img.resize((256,192))
                img = cv2.resize(img, (320,167))
                # Normalize images for training
                # img = np.array(img) / 255.0
                img = img / 255.0
                # Move RGB channel to the beginning. Needed for training models.
                img = np.moveaxis(img, 2, 0)
                _img_list.append(img)
            return (_img_list, self.label_list[lb:ub])

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

    def get_torch_dataloader(self, num_workers=1, transformation=False):
        dataset = CarlaTorchDataset(self.img_list, self.label_list, transformation)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=num_workers)

class CarlaTorchDataset(Dataset):
    def __init__(self, img_path_list, label_list, transformation=False):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transformation = transformation

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(str(self.img_path_list[idx]))
        # Crop lower part of the image to remove the car hood/roof
        img = img[:418,:,:]
        # Resize image to make it smaller
        img = cv2.resize(img, (320,167))
        # Normalize images for training
        img = img / 255.0
        # Move RGB channel to the beginning. Needed for training models.
        img = np.moveaxis(img, 2, 0)
        # Convert np array to torch tensor
        img = torch.from_numpy(img)
        # Apply transformation if needed
        if self.transformation:
            img = self.transformation(img)

        label = self.label_list[idx]

        return (img, label)