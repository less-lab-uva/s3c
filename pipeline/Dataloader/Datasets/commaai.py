from typing import Callable, Tuple

import math
import numpy as np
import random
import os

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from pipeline.Dataloader.abstract_dataloader import AbstractDataloader


class CommaAi(AbstractDataloader):
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
                assert split in ["train"], f"Sully does not have a {split} split."

        # Load images
        if input_path.is_dir():
            full_input_path = input_path/"imgFormat"
            for folder in sorted(os.listdir(full_input_path)):
                folder_path = full_input_path/folder
                if folder_path.is_dir():
                    for file in sorted(os.listdir(folder_path)):
                        if file.endswith('.jpg'):
                            img_path = folder_path/file
                            self.img_list.append(img_path)
        elif input_path.is_file():
            self.img_list.append(input_path)
        assert len(self.img_list) > 0, 'Found no images, exiting'
        # Load scene graphs if sgs_path is not None
        if sgs_path:
            if sgs_path.is_dir():
                full_sgs_path = sgs_path/"imgFormat"
                for folder in sorted(os.listdir(full_sgs_path)):
                    folder_path = full_sgs_path/folder
                    if folder_path.is_dir():
                        for sg_name in sorted(os.listdir(folder_path)):
                            if sg_name.endswith(".pkl"):
                                sg_path = folder_path/sg_name
                                self.sg_list.append(sg_path)
            elif sgs_path.is_file():
                self.sg_list.append(sgs_path)
            assert len(self.sg_list) == len(self.img_list), 'Found %d images, but %d sgs, exiting' % \
                                                            (len(self.img_list), len(self.sg_list))

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
            # Whatever needs to be done to the image for training
            _img_list.append(img)
            # Whatever needs to be done to the labels for training
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