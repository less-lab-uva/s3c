import functools
import logging
from pipeline.Dataloader.abstract_dataloader import AbstractDataloader
from pipeline.Dataloader.Datasets.cityscapes import Cityscapes
from pipeline.Dataloader.Datasets.udacity import Udacity
from pipeline.Dataloader.Datasets.nuimages import Nuimages
from pipeline.Dataloader.Datasets.sully import Sully
from pipeline.Dataloader.Datasets.commaai import CommaAi
from pipeline.Dataloader.Datasets.carla import Carla

DATALOADERS = {
    "Udacity": Udacity,
    "Nuimages": Nuimages,
    "Nuscenes": Nuimages,
    "Sully": Sully,
    "Cityscapes": Cityscapes,
    "CommaAi": CommaAi,
    'CarlaAbstract': functools.partial(Carla, type='abstract'),
    'CarlaFull': functools.partial(Carla, type='graph'),
    'CarlaRSV': functools.partial(Carla, type='rsv'),
    'CarlaFiltered': functools.partial(Carla, type='rsvfiltered'),
    'CarlaRSVSingle': functools.partial(Carla, type='rsvsingle'),
    'CarlaSingleFiltered': functools.partial(Carla, type='rsvfilteredsingle'),
    'CarlaRSVRelDists': functools.partial(Carla, type='rsvreldists'),
    'CarlaRelDistsFiltered': functools.partial(Carla, type='rsvfilteredreldists'),
    'CarlaSem': functools.partial(Carla, type='semgraph'),
    'CarlaSemRel': functools.partial(Carla, type='semgraphrel'),
    'CarlaNoRel': functools.partial(Carla, type='norelations'),
}

def DataLoaderFactory(dataset_name, dataset_path, *args, **kwargs) -> AbstractDataloader:
    logging.info('Indexing dataset, this make take several moments')
    return DATALOADERS[dataset_name](dataset_path, *args, **kwargs)
