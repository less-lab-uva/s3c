import sys  
sys.path.insert(0, '../../')

from pathlib import Path
from utils.dataset import Dataset
from pipeline.Dataloader.dataloader_factory import DataLoaderFactory

def main(arg_string):
    cluster_path = Path("../../study_data/carla_clusters/carla_rsv.json")
    data_path = Path("PATH_TO_DATA")

    # Export Town01 max_vehicles dataset in csv format
    print("Exporting Town01 max_vehicles dataset in csv format...")
    dataset_1 = Dataset.load_from_file(cluster_path, img_mnt_path=data_path, sg_mnt_path=data_path)
    dataloader_town01 = DataLoaderFactory(dataset_name="CarlaRSV", dataset_path=data_path,
                                set_splits=["Town01_max_car"], loader_type='Default', shuffle=False, save_paths_to_csv=False)
    dataset_1.filter_by_dataloader(dataloader_town01)
    dataset_1.export_as_csv(Path("./rq2/b/carla_csv/carla_max_vehicle_v3_Town01.csv"), mnt_dir=data_path, verbose=True, threads=32)

    # Export Town02 max_vehicles dataset in csv format
    print("Exporting Town02 max_vehicles dataset in csv format...")
    dataset_1 = Dataset.load_from_file(cluster_path, img_mnt_path=data_path, sg_mnt_path=data_path)
    dataloader_town02 = DataLoaderFactory(dataset_name="CarlaRSV", dataset_path=data_path,
                                set_splits=["Town02_max_car"], loader_type='Default', shuffle=False, save_paths_to_csv=False)
    dataset_1.filter_by_dataloader(dataloader_town02)
    dataset_1.export_as_csv(Path("./rq2/b/carla_csv/carla_max_vehicle_v3_Town02.csv"), mnt_dir=data_path, verbose=True, threads=32)

    # Export Town04 max_vehicles dataset in csv format
    print("Exporting Town04 max_vehicles dataset in csv format...")
    dataset_1 = Dataset.load_from_file(cluster_path, img_mnt_path=data_path, sg_mnt_path=data_path)
    dataloader_town04 = DataLoaderFactory(dataset_name="CarlaRSV", dataset_path=data_path,
                                set_splits=["Town04_max_car"], loader_type='Default', shuffle=False, save_paths_to_csv=False)
    dataset_1.filter_by_dataloader(dataloader_town04)
    dataset_1.export_as_csv(Path("./rq2/b/carla_csv/carla_max_vehicle_v3_Town04.csv"), mnt_dir=data_path, verbose=True, threads=32)

    # Export Town10HD max_vehicles dataset in csv format
    print("Exporting Town10HD max_vehicles dataset in csv format...")
    dataset_1 = Dataset.load_from_file(cluster_path, img_mnt_path=data_path, sg_mnt_path=data_path)
    dataloader_town10hd = DataLoaderFactory(dataset_name="CarlaRSV", dataset_path=data_path,
                                set_splits=["Town10HD_max_car"], loader_type='Default', shuffle=False, save_paths_to_csv=False)
    dataset_1.filter_by_dataloader(dataloader_town10hd)
    dataset_1.export_as_csv(Path("./rq2/b/carla_csv/carla_max_vehicle_v3_Town10HD.csv"), mnt_dir=data_path, verbose=True, threads=32)

    # Export full max_vehicles dataset in csv format
    print("Exporting full max_vehicles dataset in csv format...")
    dataset_1 = Dataset.load_from_file(cluster_path, img_mnt_path=data_path, sg_mnt_path=data_path)
    dataloader_1 = DataLoaderFactory(dataset_name="CarlaRSV", dataset_path=data_path,
                            set_splits=['Town01_max_car','Town02_max_car','Town04_max_car','Town10HD_max_car'], loader_type='Default', shuffle=False, save_paths_to_csv=False)
    dataset_1.filter_by_dataloader(dataloader_1)
    dataset_1.export_as_csv(Path("./rq2/b/carla_csv/carla_max_vehicle_v3.csv"), mnt_dir=data_path, verbose=True, threads=32)

if __name__ == '__main__':
    main()