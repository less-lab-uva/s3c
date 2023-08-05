# Scripts for Data Handling
This folder contains scripts for loading the data from the CARLA study and the open-source data sets.
The folder is organized as follows:
* `Dataloader`
  * `abstract_dataloader.py`: Interface for individual dataloaders. 
  * `dataloader_factory.py`: Factory class for instantiating different dataloaders.
  * `Datasets`
    * Individual Python implementations for the datasets studied:
      * `carla.py`
      * `cityscapes.py`
      * `commaai.py`
      * `nuimages.py`
      * `sully.py`
      * `udacity.py`