from typing import Tuple, Callable

import numpy as np
import random
import os

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from pipeline.Dataloader.abstract_dataloader import AbstractDataloader


class Nuimages(AbstractDataloader):
    def __init__(self, input_path, sgs_path=None, batch_size=32, shuffle=False, loader_type=None, bounds=None, *args, **kwargs) -> None:
        self.num = 0
        self.batch_size = batch_size
        self.img_list = []
        self.sg_list = []
        self.input_path = input_path
        self.sgs_path = sgs_path
        self.loader_type = "Default"
        self.strategy_types = {
            "Default":self._strategy_default,
            "Paths":self._strategy_paths
        }

        if input_path.is_dir():
            full_input_path = input_path/"sweeps/CAM_FRONT"
            for img_name in sorted(os.listdir(full_input_path)):
                if img_name.endswith(".jpg"):
                    img_path = full_input_path/img_name
                    self.img_list.append(img_path)
        elif input_path.is_file():
            self.img_list.append(input_path)
        assert len(self.img_list) > 0, 'Found no images, exiting'
        if sgs_path:
            if sgs_path.is_dir():
                full_sgs_path = sgs_path/"sweeps/CAM_FRONT"
                for sg_name in sorted(os.listdir(full_sgs_path)):
                    if sg_name.endswith(".pkl"):
                        sg_path = full_sgs_path/sg_name
                        self.sg_list.append(sg_path)
            elif sgs_path.is_file():
                self.sg_list.append(sgs_path)
            assert len(self.sg_list) == len(self.img_list), 'Found %d images, but %d sgs, exiting' % \
                                                            (len(self.img_list), len(self.sg_list))

        if bounds is not None:
            lb, ub = bounds
            self.img_list = self.img_list[lb:ub]
            if len(self.sg_list) > 0:
                self.sg_list = self.sg_list[lb:ub]
                
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

    def set_strategy(self, s_type) -> None:
        if s_type not in self.strategy_types.keys():
            raise ValueError(f"{s_type} is not a valid loader type. Use 'Default' or 'Paths.")
        else:
            if s_type == "Paths" and self.sgs_path is None:
                raise AssertionError("SGs base path missing!")
            else:
                self.loader_type = s_type