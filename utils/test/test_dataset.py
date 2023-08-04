import unittest
import tempfile
import pandas as pd

from pipeline.Dataloader.dataloader_factory import DataLoaderFactory
from pathlib import Path

from utils.dataset import Dataset


def get_udacity_loader(bounds=None):
    loader = DataLoaderFactory("Udacity",
                               Path("./pipeline/test/Datasets_sample/Udacity/"),
                               Path("./pipeline/test/SGs_sample/Udacity/"),
                               bounds=bounds,
                               loader_type="Paths")
    return loader, Dataset(loader)


def get_sully_loader():
    loader = DataLoaderFactory("Sully",
                               Path("./pipeline/test/Datasets_sample/Sully/"),
                               Path("./pipeline/test/SGs_sample/Sully/"),
                               loader_type="Paths")
    return loader, Dataset(loader)


def get_nuscenes_loader():
    loader = DataLoaderFactory("Nuscenes",
                               Path("./pipeline/test/Datasets_sample/Nuscenes/"),
                               Path("./pipeline/test/SGs_sample/Nuscenes/"),
                               loader_type="Paths")
    return loader, Dataset(loader)


def get_nuimages_loader():
    loader = DataLoaderFactory("Nuimages",
                               Path("./pipeline/test/Datasets_sample/Nuimages/"),
                               Path("./pipeline/test/SGs_sample/Nuimages/"),
                               loader_type="Paths")
    return loader, Dataset(loader)


def get_commaai_loader():
    loader = DataLoaderFactory("CommaAi",
                               Path("./pipeline/test/Datasets_sample/CommaAi/"),
                               Path("./pipeline/test/SGs_sample/CommaAi/"),
                               loader_type="Paths")
    return loader, Dataset(loader)


def get_cityscapes_loader():
    loader = DataLoaderFactory("Cityscapes",
                               Path("./pipeline/test/Datasets_sample/Cityscapes/"),
                               Path("./pipeline/test/SGs_sample/Cityscapes/"),
                               loader_type="Paths")
    return loader, Dataset(loader)


class TestDataset(unittest.TestCase):

    def test_dataset_empty_constructor(self) -> None:
        dataset = Dataset()
        self.assertEqual(0, len(dataset))

    def test_dataset_constructor(self) -> None:
        udacity_loader, udacity_dataset = get_udacity_loader()
        # in the sample, all of the SGs are unique
        self.assertEqual(len(udacity_loader), len(udacity_dataset.get_unique_scene_graphs()))
        sully_loader, sully_dataset = get_sully_loader()
        # in the sample, all of the SGs are unique
        self.assertEqual(len(sully_loader), len(sully_dataset.get_unique_scene_graphs()))

    def test_dataset_union(self) -> None:
        udacity_loader, udacity_dataset = get_udacity_loader()
        sully_loader, sully_dataset = get_sully_loader()
        udacity_union_sully = Dataset.dataset_union(udacity_dataset, sully_dataset)
        # in the sample, the intersection of the two is empty, so the union is both, and neither has unique scene graphs
        self.assertEqual(len(udacity_loader) + len(sully_loader), len(udacity_union_sully.get_unique_scene_graphs()))
        sully_union_udacity = Dataset.dataset_union(sully_dataset, udacity_dataset)
        self.assertEqual(len(udacity_loader) + len(sully_loader), len(sully_union_udacity.get_unique_scene_graphs()))
        self.assertEqual(sully_union_udacity, udacity_union_sully)

    def test_dataset_empty_intersection(self) -> None:
        udacity_loader, udacity_dataset = get_udacity_loader()
        sully_loader, sully_dataset = get_sully_loader()
        udacity_intersect_sully = Dataset.dataset_intersection(udacity_dataset, sully_dataset)
        # in the sample, the intersection of the two is empty
        self.assertEqual(0, len(udacity_intersect_sully.get_unique_scene_graphs()))
        sully_intersect_udacity = Dataset.dataset_intersection(sully_dataset, udacity_dataset)
        self.assertEqual(0, len(sully_intersect_udacity.get_unique_scene_graphs()))
        self.assertEqual(udacity_intersect_sully, sully_intersect_udacity)

    def test_dataset_nonempty_intersection(self) -> None:
        udacity_loader, udacity_dataset = get_udacity_loader()
        sully_loader, sully_dataset = get_sully_loader()
        udacity_union_sully = Dataset.dataset_union(udacity_dataset, sully_dataset)
        udacity_union_sully_intersect_udacity = Dataset.dataset_intersection(udacity_union_sully, udacity_dataset)
        self.assertEqual(udacity_dataset, udacity_union_sully_intersect_udacity)
        sully_union_udacity = Dataset.dataset_union(sully_dataset, udacity_dataset)
        sully_union_udacity_intersect_udacity = Dataset.dataset_intersection(sully_union_udacity, udacity_dataset)
        self.assertEqual(udacity_dataset, sully_union_udacity_intersect_udacity)

    def test_dataset_difference(self) -> None:
        udacity_loader, udacity_dataset = get_udacity_loader()
        sully_loader, sully_dataset = get_sully_loader()
        udacity_union_sully = Dataset.dataset_union(udacity_dataset, sully_dataset)
        udacity_union_sully_minus_sully = Dataset.dataset_difference(udacity_union_sully, sully_dataset)
        self.assertEqual(udacity_dataset, udacity_union_sully_minus_sully)
        udacity_union_sully_minus_udacity = Dataset.dataset_difference(udacity_union_sully, udacity_dataset)
        self.assertEqual(sully_dataset, udacity_union_sully_minus_udacity)
        udacity_minus_sully = Dataset.dataset_difference(udacity_dataset, sully_dataset)
        self.assertEqual(udacity_dataset, udacity_minus_sully)
        udacity_minus_udacity = Dataset.dataset_difference(udacity_dataset, udacity_dataset)
        self.assertEqual(0, len(udacity_minus_udacity))

    def test_venn(self) -> None:
        udacity_loader, udacity_dataset = get_udacity_loader()
        sully_loader, sully_dataset = get_sully_loader()
        udacity_union_sully, udacity_intersect_sully,\
        udacity_minus_sully, sully_minus_udacity = Dataset.dataset_venn(udacity_dataset, sully_dataset)
        udacity_union_sully_minus_sully = Dataset.dataset_difference(udacity_union_sully, sully_dataset)
        self.assertEqual(udacity_dataset, udacity_union_sully_minus_sully)
        udacity_union_sully_minus_udacity = Dataset.dataset_difference(udacity_union_sully, udacity_dataset)
        self.assertEqual(sully_dataset, udacity_union_sully_minus_udacity)
        udacity_union_sully_intersect_udacity = Dataset.intersect(udacity_union_sully, udacity_dataset)
        self.assertEqual(udacity_dataset, udacity_union_sully_intersect_udacity)
        udacity_union_sully_intersect_sully = Dataset.intersect(udacity_union_sully, sully_dataset)
        self.assertEqual(sully_dataset, udacity_union_sully_intersect_sully)
        # check that the symmetry holds
        sully_union_udacity, sully_intersect_udacity,\
        sully_minus_udacity2, udacity_minus_sully2 = Dataset.dataset_venn(sully_dataset, udacity_dataset)
        self.assertEqual(sully_union_udacity, udacity_union_sully)
        self.assertEqual(sully_intersect_udacity, udacity_intersect_sully)
        self.assertEqual(sully_minus_udacity, sully_minus_udacity2)
        self.assertEqual(udacity_minus_sully, udacity_minus_sully2)

    def test_dataset_clusters1(self) -> None:
        nuscenes_loader, nuscenes_dataset = get_nuscenes_loader()
        clusters = list(nuscenes_dataset.clusters())
        # there are 4 clusters
        self.assertEqual(4, len(clusters))
        # The first cluster has 2 sgs
        self.assertEqual(2, len(clusters[0][1]))
        # The rest of the clusters have 1 sg
        self.assertEqual(1, len(clusters[1][1]))
        self.assertEqual(1, len(clusters[2][1]))
        self.assertEqual(1, len(clusters[3][1]))

    def test_dataset_clusters2(self) -> None:
        commaai_loader, commaai_dataset = get_commaai_loader()
        clusters = list(commaai_dataset.clusters())
        # there are 3 clusters
        self.assertEqual(3, len(clusters))
        # The first cluster has 3 sgs (the empty sg)
        self.assertEqual(3, len(clusters[0][1]))
        # The rest of the clusters have 1 sg
        self.assertEqual(1, len(clusters[1][1]))
        self.assertEqual(1, len(clusters[2][1]))

    def test_dataset_complex_intersection(self) -> None:
        commaai_loader, commaai_dataset = get_commaai_loader()
        nuimages_loader, nuimages_dataset = get_nuimages_loader()
        commaai_intersect_nuimages = Dataset.dataset_intersection(commaai_dataset, nuimages_dataset)
        self.assertEqual(4, len(commaai_intersect_nuimages))
        clusters = list(commaai_intersect_nuimages.clusters())
        # There are 2 clusters: 3 of the empty SG and one with 1 other
        self.assertEqual(2, len(clusters))
        self.assertEqual(3, len(clusters[0][1]))
        self.assertEqual(1, len(clusters[1][1]))
        # check the reverse - intersection is not symmetric w.r.t. the image list!
        nuimages_intersect_commaai = Dataset.dataset_intersection(nuimages_dataset, commaai_dataset)
        clusters = list(nuimages_intersect_commaai.clusters())
        # There are 2 clusters: 1 of the empty SG and one with 1 other
        self.assertEqual(2, len(clusters))
        self.assertEqual(1, len(clusters[0][1]))
        self.assertEqual(1, len(clusters[1][1]))

    def test_dataset_complex_difference(self) -> None:
        commaai_loader, commaai_dataset = get_commaai_loader()
        nuimages_loader, nuimages_dataset = get_nuimages_loader()
        commaai_minus_nuimages = Dataset.dataset_difference(commaai_dataset, nuimages_dataset)
        clusters = list(commaai_minus_nuimages.clusters())
        # There is 1 cluster
        self.assertEqual(1, len(clusters))
        self.assertEqual(1, len(clusters[0][1]))
        nuimages_minus_commaai = Dataset.dataset_difference(nuimages_dataset, commaai_dataset)
        clusters = list(nuimages_minus_commaai.clusters())
        # There are 3 clusters, each with 1 image
        self.assertEqual(3, len(clusters))
        self.assertEqual(1, len(clusters[0][1]))
        self.assertEqual(1, len(clusters[1][1]))
        self.assertEqual(1, len(clusters[2][1]))

    def test_dataset_intersection_with_empty(self) -> None:
        commaai_loader, commaai_dataset = get_commaai_loader()
        empty_dataset = Dataset()
        commaai_intersect_empty = Dataset.intersect(commaai_dataset, empty_dataset)
        self.assertEqual(empty_dataset, commaai_intersect_empty)

    def test_dataset_disjoint(self) -> None:
        # Disjoint with an empty dataset
        udacity_loader, udacity_dataset = get_udacity_loader()
        empty_dataset = Dataset()
        udacity_unique, empty_unique = Dataset.dataset_disjoint(udacity_dataset, empty_dataset)
        self.assertEqual(udacity_unique, udacity_dataset)
        self.assertEqual(empty_unique, empty_dataset)
        # Disjoint between two dataset with empty intersection
        sully_loader, sully_dataset = get_sully_loader()
        udacity_unique, sully_unique = Dataset.dataset_disjoint(udacity_dataset, sully_dataset)
        self.assertEqual(udacity_unique, udacity_dataset)
        self.assertEqual(sully_unique, sully_dataset)
        # Disjoint between two datasets with nonempty intersection
        commaai_loader, commaai_dataset = get_commaai_loader()
        nuimages_loader, nuimages_dataset = get_nuimages_loader()
        commaai_intersect_nuimages = Dataset.dataset_intersection(commaai_dataset, nuimages_dataset)
        commaai_unique, nuimages_unique = Dataset.dataset_disjoint(commaai_dataset, nuimages_dataset)
        self.assertEqual(len(list(commaai_unique.clusters())), len(list(commaai_dataset.clusters())) - len(list(commaai_intersect_nuimages.clusters())))
        self.assertEqual(len(list(nuimages_unique.clusters())), len(list(nuimages_dataset.clusters())) - len(list(commaai_intersect_nuimages.clusters())))

    def test_save_to_file(self) -> None:
        udacity_loader, udacity_dataset = get_udacity_loader()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            udacity_dataset.save_to_file(temp_dir_path/"test.json", Path("pipeline/test/Datasets_sample/"), Path("pipeline/test/SGs_sample/"))
            loaded_dataset = Dataset.load_from_file(temp_dir_path/"test.json", Path("pipeline/test/Datasets_sample/"), Path("pipeline/test/SGs_sample/"))
            self.assertEqual(udacity_dataset, loaded_dataset)

    def test_filter_to_loader(self) -> None:
        udacity_loader, udacity_dataset = get_udacity_loader()
        bounded_udacity_loader, bounded_udacity_dataset = get_udacity_loader(bounds=(0, 3))
        self.assertEqual(3, len(bounded_udacity_dataset))
        udacity_dataset.filter_by_dataloader(bounded_udacity_loader)
        self.assertEqual(udacity_dataset, bounded_udacity_dataset)


    def test_export_as_csv(self) -> None:
        udacity_loader, udacity_dataset = get_udacity_loader()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            udacity_dataset.export_as_csv(temp_dir_path/"dataset.csv", "pipeline/test/Datasets_sample/")
            new_df = pd.read_csv(temp_dir_path/"dataset.csv")
            original_df = pd.read_csv("utils/test/csv_files/udacity.csv")
            self.assertEqual(new_df.equals(original_df), True)
