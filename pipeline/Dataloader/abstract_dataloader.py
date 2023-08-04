import os
from abc import ABCMeta, abstractmethod
from typing import Tuple, Callable


class AbstractDataloader(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, dataset_path:os.PathLike) -> None:
        "Init method"

    def __iter__(self):
        return self

    # Returns a list of tuples with a batch of data
    @abstractmethod
    def __next__(self) -> list:
        "__next__ method"

    # Returns the length of all the data
    @abstractmethod
    def __len__(self) -> int:
        "__len__ method"

    # Returns a tuple of lists containing the data
    @abstractmethod
    def _load(self) -> Tuple[list,list]:
        "_load method"

    # Returns a function that will load specific data
    @abstractmethod
    def _load_strategy(self) -> Callable[[int,int], Tuple[list,list]]:
        "_get_strategy"

    # Set a desired strategy on runtime
    @abstractmethod
    def set_strategy(self, s_type) -> None:
        "set_strategy"