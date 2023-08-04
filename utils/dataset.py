import time
from pathlib import Path, PosixPath
from utils.asg_compare import is_isomorphic, get_class_counts
import os
import pickle
from rustworkx import networkx_converter
from functools import lru_cache
import functools
import json
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor as ProcessPoolExecutor
from typing import Tuple, List, Dict

from tqdm import tqdm
import logging

from pipeline.Dataloader.abstract_dataloader import AbstractDataloader

import rustworkx as rx
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Node:
    def __init__(self, name, base_class=None):
        self.name = name
        self.base_class = base_class

    def __repr__(self):
        return str(self.name)


class SGUnickler(pickle.Unpickler):
    def find_class(self, module, name):
        # print(module, name)
        if (module == '__main__' or module == 'roadscene2vec.scene_graph.nodes') and name == 'Node':
            return Node
        return super().find_class(module, name)


@lru_cache(maxsize=int(os.getenv('SG_CACHE_SIZE', default='128')))
def load_sg(sg_file, convert_to_rustworkx=True):
    with open(sg_file, 'rb') as f:
        # sg = pickle.load(f)
        sg = SGUnickler(f).load()
    # convert from networkx to rustworkx for efficiency
    if convert_to_rustworkx:
        sg = networkx_converter(sg)
    return sg


def is_empty_sg(sg) -> bool:
    if isinstance(sg, str) or isinstance(sg, PosixPath):
        sg = load_sg(sg)
    return is_isomorphic(sg, EMPTY_SG)


def get_sg_metadata(sg, bypass_cache=False) -> Tuple[int, int]:
    if isinstance(sg, str) or isinstance(sg, PosixPath):
        if bypass_cache:
            sg = load_sg.__wrapped__(sg)
        else:
            sg = load_sg(sg)
    return sg.num_nodes(), sg.num_edges(), get_class_counts(sg)


def find_iso(sg1_file, cluster_keys, sg_files,
             check_preconditions: bool = True, timeout: float = -1, verbose: bool = False, leave: bool = True):
    """Find the cluster that the SG given by sg1_file belongs in, i.e. is isomorphic to"""
    sg1 = load_sg(sg1_file)
    with tqdm(total=len(cluster_keys), disable=not verbose, leave=leave) as pbar:
        for key2 in cluster_keys:
            sg2 = load_sg(sg_files[key2])
            if is_isomorphic(sg1, sg2, timeout=timeout, check_preconditions=check_preconditions):
                pbar.update(len(cluster_keys))
                # return the key as soon as we find it
                return key2
            pbar.update(1)
    # if we have not found a key after looking through all of them, return None
    return None


EMPTY_SG = load_sg(Path(__file__).parent / 'empty_sg.pkl')
EMPTY_SG_METADATA = get_sg_metadata(EMPTY_SG)


class IsoTimeoutError(Exception):
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def __str__(self):
        return f'"{self.file1}", "{self.file2}"'

    def get_files(self):
        return self.file1, self.file2


def try_all_cluster_candidates(cur_sg_file, cur_sg, cluster, sg_map, timeout=-1, threads=-1):
    """
    Try the current SG against all SGs in the cluster for isomorphism.

    This may be useful if the representation of the SG happens to make it particularly hard to compare
    """
    # if the cluster is empty, then we are not isomorphic to the cluster. This shouldn't be called with an empty
    # cluster, but we should protect against that here
    if len(cluster) == 0:
        return False
    if cur_sg is None:
        cur_sg = load_sg(cur_sg_file)
    if threads < 0:
        # Threads < 0 really means single-threaded without threading overhead
        # Check each of them in order, considering the timeout
        for other_image_file in cluster:
            other_sg = load_sg(sg_map[other_image_file])
            # this will try for at most timeout seconds. If timeout is -1, it will run until an answer is found
            is_iso = is_isomorphic(cur_sg, other_sg, timeout=timeout, check_preconditions=False)
            # if the timeout was reached, then None is returned, and we should proceed to the next one in the list
            if is_iso is not None:
                # we managed to find a concrete answer without timing out
                return is_iso
    else:
        # This block is not used by default, and current code does not call it.
        # The goal of this pool is to check all of the candidates in parallel, up to the timeout.
        # This is sound because all of the graphs in the list are guaranteed to be isomorphic to each other,
        # so if the candidate graph is isomorphic to one of them, then it is isomorphic to all of them, and
        # by contrast if it is not isomorphic to one of them, then it is not isomorphic to any of them.
        # If the number of threads is less than the number of candidates, only the first candidates up to the number
        # of threads are checked.
        # The below creates a list of sgs that the pool will check
        cluster_sgs = [sg_map[other_image_file] for other_image_file in cluster]
        with Pool(threads) as pool:
            # This dispatches the full list to the pool. The iterator will return the first one to finish,
            # not the first one in the list.
            result_iterator = pool.imap_unordered(functools.partial(is_isomorphic,
                                                                    asg2=cur_sg,
                                                                    check_preconditions=False),
                                                  cluster_sgs)
            if timeout < 0:
                # if there is no timeout, wait for the first one that finishes, and then report that result
                return result_iterator.next()
            else:
                try:
                    # if there is a timeout, wait either the timeout to elapse or the first one to finish
                    return result_iterator.next(timeout)  # we got an answer, return True or False
                except multiprocessing.context.TimeoutError:
                    # if the timeout elapses, then return None to signal that we don't know
                    return None

    # if we got this far without returning, then all of them timed out
    return None


def get_clusters(image_files: List[str], sg_map: Dict[str, str],
                 log=False, timeout=-1, drop_after_timeout=False, max_try_again=0, threads=-1) -> List[List[str]]:
    clusters, could_not_cluster = __get_clusters_timed(image_files, log, sg_map, timeout, threads=threads)
    if log and len(could_not_cluster) > 0:
        if log:
            logging.info('After first round, %d clusters found and %d images could not be clustered within the timeout'
                         % (len(clusters), len(could_not_cluster)))
        if not drop_after_timeout:
            logging.info('Retrying with no timeout as drop_after_timeout=False')
            # first, try again with double timeout
            prev_could_not_cluster = len(could_not_cluster)
            check_again = True
            count = 0
            # We will try again a few times to see if we have found a new SG that we can check faster
            # We will stop if we reach a fixed point or run out of tries.
            # Note, it has not been empirically verified if this improves performance - it is possible that this is only
            # duplicating previous work. However, it is sound, and theoretically there may be times when this would
            # render better performance.
            while check_again and count < max_try_again:
                clusters, could_not_cluster = __get_clusters_timed([image_file for image_file, _ in could_not_cluster],
                                                                   log, sg_map, timeout * 2, threads=threads,
                                                                   existing_clusters=clusters)
                check_again = len(could_not_cluster) < prev_could_not_cluster
                prev_could_not_cluster = len(could_not_cluster)
            # if we are still missing some, try again with no timeout
            for image_file, _ in tqdm(could_not_cluster, disable=not log):  # , leave=False
                cur_sg = load_sg(sg_map[image_file])
                found_cluster = False
                # check the current SG against the existing clusters
                for cluster in clusters:
                    # use the back of the cluster bc we probably didn't try that one before, and it might be better
                    other_sg = load_sg(sg_map[cluster[-1]])
                    is_iso = is_isomorphic(cur_sg, other_sg, check_preconditions=False)
                    assert (is_iso is not None)  # if we got a None here, something failed
                    if is_iso:
                        found_cluster = True
                        cluster.append(image_file)
                        break
                if not found_cluster:
                    clusters.append([image_file])
    if log:
        logging.info('Found %d clusters for %d images (%0.2f%%)' %
                     (len(clusters), len(image_files), len(clusters) / len(image_files)))
    return clusters


def __get_clusters_timed(image_files, log, sg_map, timeout, threads=-1, existing_clusters=None):
    clusters = [] if existing_clusters is None else existing_clusters
    could_not_cluster = []
    for image_file in tqdm(image_files, disable=not log):  # , leave=False
        try:
            cur_sg_file = sg_map[image_file]
            cur_sg = load_sg(cur_sg_file)
            found_cluster = False
            # check the current SG against the existing clusters
            for cluster in clusters:
                is_iso = try_all_cluster_candidates(cur_sg_file, cur_sg, cluster, sg_map, timeout, threads)
                if is_iso is None:
                    raise IsoTimeoutError(image_file, cluster[0])
                if is_iso:
                    found_cluster = True
                    cluster.append(image_file)
                    break
            # if we were unable to find a cluster to add it to, make a cluster of just this image
            if not found_cluster:
                clusters.append([image_file])
        except IsoTimeoutError as iso_error:
            could_not_cluster.append(iso_error.get_files())
    return clusters, could_not_cluster


def _merge_clusters(clusterings, sg_map):
    if len(clusterings) == 0:
        return clusterings
    # start by copying the image lists from the first clustering
    final_clusters = [cluster_images for cluster_images in clusterings[0]]
    # for the rest of the clusterings, go through one by one and check if they can be merged into the final
    for clustering in clusterings[1:]:
        for cluster in clustering:
            cur_sg = load_sg(sg_map[cluster[0]])
            found_match = False
            for final_cluster in final_clusters:
                # if the two representatives are isomorphic, then all are isomorphic, merge
                if is_isomorphic(cur_sg, load_sg(sg_map[final_cluster[0]]), check_preconditions=False):
                    final_cluster.extend(cluster)
                    found_match = True
                    break
            # if we did not find a match, then it was not isomorphic to any of them, so we need to add it to the final
            if not found_match:
                final_clusters.append(cluster)
    return final_clusters


class Dataset:
    def __init__(self, dataloader: AbstractDataloader = None, threads=1, max_per_thead=512, load_threaded=False):
        # map from image file to sg
        self._sg_files = {}
        # list of image files
        self._image_files = []
        # map from image_file to list of image files
        self._clusters = {}
        # list of image_files that are keys to _clusters in sorted order
        self._sorted_cluster_keys = []
        self._dataloader = None
        self._has_clusters = False
        self._graph_metadata_groups = {}
        self._image_to_metadata = {}
        if dataloader is not None:
            self._load_from_dataloader(dataloader)
            self._gen_clusters(threads=threads, max_per_thread=max_per_thead, load_threaded=load_threaded)

    def _serialize_sg_files(self, img_mnt_path, sg_mnt_path):
        # Convert Path of sg_files to Str using relative paths w.r.t. mnt points
        sg_files = {}
        for key, val in self._sg_files.items():
            str_key = os.path.relpath(key, img_mnt_path)
            str_val = os.path.relpath(val, sg_mnt_path)
            sg_files[str_key] = str_val
        return sg_files

    def _serialize_image_files(self, img_mnt_path):
        # Convert Path of image_files to Str using relative paths w.r.t. mnt point
        image_files = []
        for val in self._image_files:
            str_val = os.path.relpath(val, img_mnt_path)
            image_files.append(str_val)
        return image_files

    def _serialize_clusters(self, img_mnt_path):
        # Convert Path of clusters to Str using relative paths w.r.t. mnt point
        clusters = {}
        for key, val_list in self._clusters.items():
            str_key = os.path.relpath(key, img_mnt_path)
            str_list = []
            for val in val_list:
                str_val = os.path.relpath(val, img_mnt_path)
                str_list.append(str_val)
            clusters[str_key] = str_list
        return clusters

    def _serialize_graph_metadata_groups(self, img_mnt_path):
        # convert Path of groupings by number of nodes/edges to relative paths w.r.t. mnt point
        meta_groups = {}
        for size, val_list in self._graph_metadata_groups.items():
            str_list = []
            for val in val_list:
                str_val = os.path.relpath(val, img_mnt_path)
                str_list.append(str_val)
            meta_groups[str(size)] = str_list
        return meta_groups

    def save_to_file(self, filename, img_mnt_path, sg_mnt_path):
        """Saves the dataset as a json using the files as keys"""
        json_obj = {
            'sg_files': self._serialize_sg_files(img_mnt_path, sg_mnt_path),
            'image_files': self._serialize_image_files(img_mnt_path),
            'clusters': self._serialize_clusters(img_mnt_path),
            'metadata_groups': self._serialize_graph_metadata_groups(img_mnt_path),
            'version': 1
        }
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w+') as f:
            json.dump(json_obj, f)

    @staticmethod
    def __deserialize_sg_files(json_sg_files, img_mnt_path, sg_mnt_path):
        # Convert Str of sg_files to Path using absolute path with mnt points
        sg_files = {}
        for key, val in json_sg_files.items():
            path_key = img_mnt_path / Path(key)
            path_val = sg_mnt_path / Path(val)
            sg_files[path_key] = path_val
        return sg_files

    @staticmethod
    def __deserialize_image_files(json_image_files, img_mnt_path):
        # Convert Str of image_files to Path using absolute path with mnt point
        image_files = []
        for val in json_image_files:
            path_val = img_mnt_path / Path(val)
            image_files.append(path_val)
        return image_files

    @staticmethod
    def __deserialize_clusters(json_clusters, img_mnt_path):
        # Convert Str of clusters to Path using absolute path with mnt point
        clusters = {}
        for key, val_list in json_clusters.items():
            path_key = img_mnt_path / Path(key)
            path_list = []
            for val in val_list:
                path_val = img_mnt_path / Path(val)
                path_list.append(path_val)
            clusters[path_key] = path_list
        return clusters

    @staticmethod
    def __deserialize_metadata_groups(json_node_edge_groups, img_mnt_path):
        # Convert Str of images grouped by number of nodes and edges to Path using absolute path with mnt point
        metadata_groups = {}
        for metadata, val_list in json_node_edge_groups.items():
            path_list = []
            for val in val_list:
                path_val = img_mnt_path / Path(val)
                path_list.append(path_val)
            # the metadata is a str representation of a tuple of ints and strings, so evaluate it
            metadata_groups[eval(metadata)] = path_list
        return metadata_groups

    @classmethod
    def load_from_file(cls, filename, img_mnt_path=None, sg_mnt_path=None):
        if img_mnt_path is None:
            img_mnt_path = Path()
        if sg_mnt_path is None:
            sg_mnt_path = Path()
        dataset = cls()
        with open(filename, 'r') as f:
            json_obj = json.load(f)
            dataset._sg_files = Dataset.__deserialize_sg_files(json_obj['sg_files'], img_mnt_path, sg_mnt_path)
            dataset._image_files = Dataset.__deserialize_image_files(json_obj['image_files'], img_mnt_path)
            dataset._clusters = Dataset.__deserialize_clusters(json_obj['clusters'], img_mnt_path)
            dataset._graph_metadata_groups = Dataset.__deserialize_metadata_groups(json_obj['metadata_groups'],
                                                                                   img_mnt_path)
            for size, image_file_list in dataset._graph_metadata_groups.items():
                for image_file in image_file_list:
                    dataset._image_to_metadata[image_file] = size
        dataset._has_clusters = dataset._clusters is not None and len(dataset._clusters) > 0
        dataset._sorted_cluster_keys = dataset.__get_sorted_cluster_keys()
        return dataset

    def get_unique_scene_graphs(self, load_sgs=False):
        """Returns the list of unique SGs, i.e. one for each cluster"""
        sg_files = [self._sg_files[cluster_key] for cluster_key in self._sorted_cluster_keys]
        return [load_sg(sg_file) for sg_file in sg_files] if load_sgs else sg_files

    def _load_from_dataloader(self, dataloader: AbstractDataloader = None):
        self._dataloader = dataloader
        # the data loader returns the files in batches
        for image_files, sg_files in dataloader:
            # for each batch, pull out the files
            for image_file, sg_file in zip(image_files, sg_files):
                self._sg_files[image_file] = sg_file
                self._image_files.append(image_file)
        logging.info('Dataset loaded %d image/sg pairs' % len(self._image_files))

    def __len__(self):
        return len(self._image_files)

    def _index_sg_metadata(self, metadata, image_file):
        """
        Maintains an index of the image based on its number of nodes and edges.

        This is used to get the list of images with the same size, as well as to cache the size for later
        """
        if metadata not in self._graph_metadata_groups:
            self._graph_metadata_groups[metadata] = []
        self._graph_metadata_groups[metadata].append(image_file)
        self._image_to_metadata[image_file] = metadata

    def get_sizing_groups(self):
        return dict(self._graph_metadata_groups)

    def image_to_cluster_map(self):
        image_to_cluster = {}
        for cluster_key, cluster in self._clusters.items():
            for image in cluster:
                image_to_cluster[image] = cluster_key
        assert len(image_to_cluster) == len(self._image_files)
        return image_to_cluster

    def _gen_clusters(self, threads=1, max_per_thread=512, load_threaded=False):
        tic = time.perf_counter_ns()
        logging.info('Generating clusters for dataset using %d threads' % threads)
        self._has_clusters = True
        if not load_threaded or threads <= 1:
            logging.info('Performing initial size indexing to speed up clustering')
            for image_file in tqdm(self._image_files):
                metadata = get_sg_metadata(self._sg_files[image_file], bypass_cache=True)
                self._index_sg_metadata(metadata, image_file)
        if threads > 1:
            with ProcessPoolExecutor(threads) as pool:
                if load_threaded:
                    logging.info('Performing initial size indexing to speed up clustering')
                    size_results = {}
                    for image_file in self._image_files:
                        size_results[image_file] = pool.submit(get_sg_metadata, self._sg_files[image_file])
                    for image_file, size_result in tqdm(size_results.items()):
                        metadata = size_result.result()
                        self._index_sg_metadata(metadata, image_file)
                logging.info('Performing clustering')
                results = {-1: []}
                auto_cluster_count = 0
                # iterating in sorted order allows us to show some progress to the user at first
                sorted_sizes = sorted(self._graph_metadata_groups.keys(),
                                      key=lambda x: len(self._graph_metadata_groups[x]))
                groups_remaining = []
                regroup_group = 0
                logging.info('Dispatching jobs to threads')
                for size in tqdm(sorted_sizes):
                    group = self._graph_metadata_groups[size]
                    # if there is only one graph with this size, then it is its own cluster
                    # similarly, the empty graph is the smallest graph, and under the roadscene2vec paradigm, is unique
                    # thus, if the group has the same size as the empty sg, then it is its own cluster
                    if len(group) == 1 or size == EMPTY_SG_METADATA:
                        self._clusters[group[0]] = group
                        auto_cluster_count += len(group)
                    else:
                        groups_remaining.append(len(group))
                        # if max_per_thread is -1, then don't split
                        # otherwise, if the length of the group is greater than max_per_thread, then split
                        # the group into smaller chunks that we will then merge later
                        if len(group) > max_per_thread != -1:
                            results[regroup_group] = []
                            for start in range(0, len(group), max_per_thread):
                                end = min(start + max_per_thread, len(group))
                                subgroup = group[start:end]
                                results[regroup_group].append(
                                    pool.submit(get_clusters, subgroup,
                                                self._sg_files,
                                                False,  # do not log
                                                5,  # 5 second timeout
                                                False,  # do not give up on those exceeding timeout
                                                2,  # number of retries before waiting out isomorphism
                                                -1  # number of threads to multiprocess. -1 means one at a time
                                                ))
                            regroup_group += 1
                        else:
                            results[-1].append(pool.submit(get_clusters, group, self._sg_files,
                                                           False,  # do not log
                                                           5,  # 5 second timeout
                                                           False,  # do not give up on those exceeding timeout
                                                           2,  # number of retries before waiting out isomorphism
                                                           -1
                                                           # number of threads to multiprocess. -1 means one at a time
                                                           ))
                logging.info('After size sorting, %d automatically clustered into %d clusters.'
                             ' Checking remaining graphs. Progress may be very inconsistent'
                             % (auto_cluster_count, len(self._clusters)))
                # using tqdm this way allows us to keep up with things as they finish rather than
                # having one bottleneck reporting the progress of the rest. This also helps with throughput and memory
                # because the finished results can be processed ahead of the others.
                logging.info('There are %d groups remaining to cluster, group sizes are:' % len(groups_remaining))
                logging.info(str(groups_remaining))
                cluster_merge_results = []
                result_checked = set()
                group_done = set()
                merge_results_done = set()
                with tqdm(total=sum(groups_remaining)) as progress:
                    all_done = False
                    while not all_done:
                        all_done = True
                        time.sleep(1)
                        for regroup_group, result_list in results.items():
                            if regroup_group == -1:
                                # -1 is a flag for items that were not split up and thus don't need to be merged
                                for result_index, result in enumerate(result_list):
                                    if result.done():
                                        if result_index not in result_checked:
                                            result_checked.add(result_index)
                                            clusters = result.result()
                                            update_total = 0
                                            for cluster in clusters:
                                                update_total += len(cluster)
                                                self._clusters[cluster[0]] = cluster
                                            progress.update(update_total)
                                    else:
                                        all_done = False
                            else:
                                # otherwise, check if every part of the group is done and thus ready to be merged
                                if regroup_group not in group_done:
                                    group_finished = True
                                    for result in result_list:
                                        if not result.done():
                                            # if any one of them is not done, then we cannot merge
                                            group_finished = False
                                            all_done = False
                                            break
                                    if group_finished:
                                        # if the group is finished, then we can merge everything together
                                        # do this in another thread so that we can keep going
                                        group_done.add(regroup_group)
                                        # merge results back
                                        intermediate_clusterings = [result.result() for result in result_list]
                                        cluster_merge_results.append(pool.submit(_merge_clusters,
                                                                                 intermediate_clusterings,
                                                                                 self._sg_files))
                        for merge_index, cluster_merge_result in enumerate(cluster_merge_results):
                            if merge_index not in merge_results_done:
                                if cluster_merge_result.done():
                                    merge_results_done.add(merge_index)
                                    clusters = cluster_merge_result.result()
                                    update_total = 0
                                    for cluster in clusters:
                                        update_total += len(cluster)
                                        self._clusters[cluster[0]] = cluster
                                    logging.info('updating from merge %d' % update_total)
                                    progress.update(update_total)
                                else:
                                    all_done = False
        else:
            logging.info('Calculating clusters one at a time. Progress may be very inconsistent.')
            # iterating in sorted order allows us to show some progress to the user at first
            sorted_sizes = sorted(self._graph_metadata_groups.keys(), key=lambda x: len(self._graph_metadata_groups[x]))
            for size in sorted_sizes:
                group = self._graph_metadata_groups[size]
                if len(group) == 1 or size == EMPTY_SG_METADATA:
                    self._clusters[group[0]] = group
                else:
                    clusters = get_clusters(group, self._sg_files, log=True)
                    for cluster in clusters:
                        self._clusters[cluster[0]] = cluster
        self._sorted_cluster_keys = self.__get_sorted_cluster_keys()
        toc = time.perf_counter_ns()
        logging.info('Found %d clusters for %d image/sg pairs in %0.2f seconds' %
                     (len(self._sorted_cluster_keys), len(self._image_files), 1e-9 * (toc - tic)))

    def __get_sorted_cluster_keys(self):
        new_clusters = {}
        # normalize the clusters so that they key is the lex lowest image, and the images are lex sorted
        # this allows for efficient comparison in __eq__
        for cluster_key, image_files in self._clusters.items():
            sorted_image_files = sorted(image_files)
            new_clusters[sorted_image_files[0]] = sorted_image_files
        self._clusters = new_clusters
        return sorted(self._clusters.keys(), key=lambda img_file: len(self._clusters[img_file]), reverse=True)

    def get_sorted_cluster_keys(self):
        return list(self._sorted_cluster_keys)

    def get_cluster(self, cluster_key):
        return list(self._clusters[cluster_key])

    def __eq__(self, other):
        if not isinstance(other, Dataset):
            return False
        # validate that images are the same
        if set(self._image_files) != set(other._image_files):
            return False
        # validate that sgs are the same
        if set(self._sg_files) != set(other._sg_files):
            return False
        # check clusters - they should be maintained in canonical form by __get_sorted_cluster_keys
        if self._clusters != other._clusters:
            return False
        # If the clusters are correct then this should be definitionally correct, but add it to the eq check
        # so that the tests make sure that we enforce our invariants appropriately
        if self._graph_metadata_groups != other._graph_metadata_groups:
            return False
        if self._image_to_metadata != other._image_to_metadata:
            return False
        return True

    def clusters(self, load_sgs=False):
        """
        yields sg, list of images where each SG is the cluster representative and the list of associated image files

        Clusters are returned in sorted order from largest to smallest
        """
        for cluster_key in self._sorted_cluster_keys:
            if load_sgs:
                yield load_sg(self._sg_files[cluster_key]), self._clusters[cluster_key]
            else:
                yield self._sg_files[cluster_key], self._clusters[cluster_key]

    @staticmethod
    def __copy_dataset(src, dst, clusters_to_keep=None, src_to_dst_map=None):
        """
        Modifies dst in place, copying the clusters and all associated data from src.

        Either the intersection between src and dest must be empty, or src_to_dst_map must be provided containing
        a mapping from keys in src to keys in dest that are isomorphic.
        """
        # iterate over the clusters to keep and add them to the dataset
        dst._has_clusters = True
        updated_images = []
        if clusters_to_keep is None:
            # if None, default to all
            clusters_to_keep = src._sorted_cluster_keys
        for src_cluster_key in clusters_to_keep:
            cluster_image_files = src._clusters[src_cluster_key]
            dst_key = None
            if src_to_dst_map is not None and src_cluster_key in src_to_dst_map:
                # pull the matched key so that we extend that cluster
                dst_key = src_to_dst_map[src_cluster_key]
            if dst_key is None:
                # we weren't able to find a match, so fall back on the original
                dst_key = src_cluster_key
            if dst_key in dst._clusters:
                dst._clusters[dst_key].extend(cluster_image_files)
            else:
                dst._clusters[dst_key] = cluster_image_files
            updated_images.extend(cluster_image_files)
        dst._image_files.extend(updated_images)
        # once the list of images has been extracted from the clusters, use that to add the sgs and images
        for image_file in updated_images:
            if image_file in dst._sg_files:
                continue
            dst._sg_files[image_file] = src._sg_files[image_file]
            metadata = src._image_to_metadata[image_file]
            dst._index_sg_metadata(metadata, image_file)
        # if dst was not empty, the sort order could have changed
        dst._sorted_cluster_keys = dst.__get_sorted_cluster_keys()

    def union(self, other):
        return Dataset.dataset_union(self, other)

    def intersect(self, other):
        return Dataset.dataset_intersection(self, other)

    def minus(self, other):
        return Dataset.dataset_difference(self, other)

    def diff(self, other):
        return self.minus(other)

    def filter_by_image_list(self, image_files):
        current_list = set(self._image_files)
        dataloader_list = set(image_files)
        for cluster_key in self._sorted_cluster_keys:
            # reduce the cluster to only those that are in the dataloader
            cluster = [retained_image for retained_image in self._clusters[cluster_key]
                       if retained_image in dataloader_list]
            # delete the original cluster
            del self._clusters[cluster_key]
            # if there are any left, re-add the cluster, choosing a new representative.
            # This will be re-normalized at the end
            if len(cluster) > 0:
                self._clusters[cluster[0]] = cluster
        # need to wrap the keys in a list because it will change size in the loop
        for metadata in list(self._graph_metadata_groups.keys()):
            # reduce the group to only those that are in the dataloader
            new_metadata_group = [retained_image for retained_image in self._graph_metadata_groups[metadata]
                                  if retained_image in dataloader_list]
            if len(new_metadata_group) == 0:
                # if there are none left, then delete the entry
                del self._graph_metadata_groups[metadata]
            else:
                # otherwise, update the list
                self._graph_metadata_groups[metadata] = new_metadata_group
        diff = current_list.difference(dataloader_list)
        for image_path in diff:
            # Delete sg_file
            del self._sg_files[image_path]
            # Delete the metadata
            del self._image_to_metadata[image_path]
        # Replace the _image_files list with the one contained in the dataloader
        self._image_files = list(image_files)
        # This will handle re-normalizing the clusters as well
        self._sorted_cluster_keys = self.__get_sorted_cluster_keys()
        self._dataloader = None  # unset - if called by filter_by_dataloader, it will reset it.

    def filter_by_dataloader(self, dataloader):
        """Filters the dataset in place"""
        self.filter_by_image_list(dataloader.img_list)
        self._dataloader = dataloader

    @classmethod
    def dataset_difference(cls, minuend: 'Dataset', subtrahend: 'Dataset'):
        """Returns the result of minuend \\ subtrahend as a new dataset object"""
        clusters_to_keep = [minuend_cluster_key for minuend_cluster_key in minuend._sorted_cluster_keys]
        # iterate over things in the subtrahend and remove them if they are in the minuend
        for subtrahend_cluster_key in subtrahend._sorted_cluster_keys:
            subtrahend_sg = load_sg(subtrahend._sg_files[subtrahend_cluster_key])
            for minuend_cluster_key in minuend._sorted_cluster_keys:
                minuend_sg = load_sg(minuend._sg_files[minuend_cluster_key])
                if is_isomorphic(subtrahend_sg, minuend_sg):
                    clusters_to_keep.remove(minuend_cluster_key)
                    break
        difference_dataset = cls()
        Dataset.__copy_dataset(minuend, difference_dataset, clusters_to_keep)
        return difference_dataset

    @classmethod
    def dataset_intersection(cls, dataset1: 'Dataset', dataset2: 'Dataset'):
        """
        Returns the result of dataset1 ∩ dataset2 as a new dataset object

        THIS IS NOT SYMMETRIC w.r.t. the image list and clusters!
        A ∩ B and B ∩ A will have the same set of scene graphs, but different clusters.
        A ∩ B takes the clusters of A and filters them to only retain those with SGs in B.
        Accordingly, this does not add the representatives of those SGs from B into the intersection.
        """
        # there could be efficiency improvements by rearranging which is dataset1/2 by size
        clusters_to_keep = [dataset1_key for dataset1_key in dataset1._sorted_cluster_keys]
        # iterate over things in dataset1 and remove them if they are _not_ in dataset 2
        for dataset1_key in dataset1._sorted_cluster_keys:
            dataset1_sg = load_sg(dataset1._sg_files[dataset1_key])
            # track to see if the dataset1 sg has a match in dataset2
            has_match = False
            for dataset2_key in dataset2._sorted_cluster_keys:
                dataset2_sg = load_sg(dataset2._sg_files[dataset2_key])
                if is_isomorphic(dataset1_sg, dataset2_sg):
                    # if we have found the match, mark that and stop looking
                    has_match = True
                    break
            # if no match was found, it is only in one data set and should be removed
            if not has_match:
                clusters_to_keep.remove(dataset1_key)
        intersection_dataset = cls()
        Dataset.__copy_dataset(dataset1, intersection_dataset, clusters_to_keep)
        return intersection_dataset

    @classmethod
    def dataset_venn(cls, dataset1: 'Dataset', dataset2: 'Dataset', threads: int = 1,
                     timeout: 'float' = -1.0, verbose: 'bool' = False):
        """
        Returns a tuple of 4 datasets representing:
         (dataset1 ∪ dataset2, dataset1 ∩ dataset2, dataset1 \\ dataset2, dataset2 \\ dataset1)
         """
        dataset_swap = False
        if len(dataset2._sorted_cluster_keys) > len(dataset1._sorted_cluster_keys):
            # make dataset 2 have fewer clusters than dataset 1.
            # This may help improve performance by increasing the parallelization and decreasing work per thread
            temp = dataset1
            dataset1 = dataset2
            dataset2 = temp
            dataset_swap = True  # note that we swapped it so we can put it back when we return
        union_dataset = cls()
        intersection_dataset = cls()
        d1_diff_d2 = cls()
        d2_diff_d1 = cls()
        d1_key_map = {}
        d2_matched = []
        unmatched_d1_keys = []
        if threads == 1:
            if verbose:
                logging.info(f"Finding dataset matching single-threaded. Will report fine-grained progress.")
            for d1_key in tqdm(dataset1._sorted_cluster_keys, disable=not verbose):
                result = None
                # we can use the metadata groups to pre-filter the iso check
                # we can additionally filter that list based on ones that have already been matched
                d1_meta_group = dataset1._image_to_metadata[d1_key]
                if d1_meta_group in dataset2._graph_metadata_groups:
                    filtered_list = [d2_key for d2_key in
                                     dataset2._graph_metadata_groups[d1_meta_group]
                                     if d2_key not in d2_matched]
                    result = find_iso(dataset1._sg_files[d1_key], filtered_list, dataset2._sg_files,
                                      False, timeout, verbose=verbose, leave=False)
                if result is not None:
                    d1_key_map[d1_key] = result
                    d2_matched.append(result)
                else:
                    unmatched_d1_keys.append(d1_key)
        else:
            if verbose:
                logging.info(f"Finding dataset matching with {threads} threads. Will only report total progress.")
            with ProcessPoolExecutor(threads) as pool:
                results = {}
                for d1_key in dataset1._sorted_cluster_keys:
                    d1_meta_group = dataset1._image_to_metadata[d1_key]
                    if d1_meta_group in dataset2._graph_metadata_groups:
                        results[d1_key] = pool.submit(find_iso, dataset1._sg_files[d1_key],
                                                      # we can use the metadata groups to pre-filter the iso check
                                                      dataset2._graph_metadata_groups[dataset1._image_to_metadata
                                                                                      [d1_key]],
                                                      dataset2._sg_files, False, timeout)
                    else:
                        results[d1_key] = None
                for d1_key, result in tqdm(results.items(), disable=not verbose):
                    # result could already be none, has same semantics as result.result() being None
                    if result is not None:
                        result = result.result()
                    if result is not None:
                        d1_key_map[d1_key] = result
                        d2_matched.append(result)
                    else:
                        unmatched_d1_keys.append(d1_key)
        unmatched_d2_keys = [d2_key for d2_key in dataset2._sorted_cluster_keys if d2_key not in d2_matched]
        # we now have:
        # unmatched_d1_keys with a list of what goes in D1 \ D2
        # unmatched_d2_keys with a list of what goes in D2 \ D1
        # d1_key_map showing the mapping between keys for those in the intersection

        # start by extracting the differences using the unmatched keys
        Dataset.__copy_dataset(dataset1, d1_diff_d2, unmatched_d1_keys)
        Dataset.__copy_dataset(dataset2, d2_diff_d1, unmatched_d2_keys)

        # then use the mapping to create the intersection
        # start by copying in the portion from dataset2
        Dataset.__copy_dataset(dataset2, intersection_dataset, d1_key_map.values())
        # then copy in the part from dataset2
        Dataset.__copy_dataset(dataset1, intersection_dataset, d1_key_map.keys(), d1_key_map)

        # Now the union is exactly the two differences plus the intersection
        Dataset.__copy_dataset(d1_diff_d2, union_dataset)
        Dataset.__copy_dataset(d2_diff_d1, union_dataset)
        Dataset.__copy_dataset(intersection_dataset, union_dataset)

        if dataset_swap:
            # if we swapped at the beginning, we need to swap back
            temp = d1_diff_d2
            d1_diff_d2 = d2_diff_d1
            d2_diff_d1 = temp
        return union_dataset, intersection_dataset, d1_diff_d2, d2_diff_d1

    @classmethod
    def dataset_union(cls, dataset1: 'Dataset', dataset2: 'Dataset',
                      timeout: 'float' = -1, verbose: 'bool' = False):
        """Returns the result of dataset1 ∪ dataset2 as a new dataset object"""
        # there could be efficiency improvements by rearranging which is dataset1/2 by size
        union_dataset = cls()
        # start by copying all  of dataset1 into the union
        Dataset.__copy_dataset(dataset1, union_dataset, dataset1._clusters.keys())
        # keep track of which ones have not been matched from dataset2 so we can add them at the end
        unmatched_dataset2_clusters = [dataset2_key for dataset2_key in dataset2._sorted_cluster_keys]
        # iterate through the union clusters and merge dataset2 clusters into them as needed
        for union_key in tqdm(union_dataset._sorted_cluster_keys, disable=not verbose):
            union_sg = load_sg(union_dataset._sg_files[union_key])
            for dataset2_key in tqdm(unmatched_dataset2_clusters, leave=False, disable=not verbose):
                dataset2_sg = load_sg(dataset2._sg_files[dataset2_key])
                # if we found a match between dataset2 and the union, merge those clusters
                if is_isomorphic(union_sg, dataset2_sg, timeout):
                    unmatched_dataset2_clusters.remove(dataset2_key)
                    # merge the dataset2 images from the dataset2 cluster into the union cluster
                    dataset2_image_files_to_add = dataset2._clusters[dataset2_key]
                    union_dataset._clusters[union_key].extend(dataset2_image_files_to_add)
                    union_dataset._image_files.extend(dataset2_image_files_to_add)
                    for dataset2_image_file in dataset2_image_files_to_add:
                        union_dataset._sg_files[dataset2_image_file] = dataset2._sg_files[dataset2_image_file]
                        metadata = dataset2._image_to_metadata[dataset2_image_file]
                        union_dataset._index_sg_metadata(metadata, dataset2_image_file)
                    break
            if len(unmatched_dataset2_clusters) == 0:
                break  # in case dataset 2 is a subset of dataset1, break early
        # once we have merged everything we can, all that remains are sgs that are only in dataset2
        # use the copy function to bulk copy those into the union
        Dataset.__copy_dataset(dataset2, union_dataset, unmatched_dataset2_clusters)
        return union_dataset

    @classmethod
    def dataset_disjoint(cls, dataset1: 'Dataset', dataset2: 'Dataset'):
        """Returns the result of (dataset1 \\ dataset2) and (dataset2 \\ dataset1) as two new dataset objects"""
        return Dataset.dataset_difference(dataset1, dataset2), Dataset.dataset_difference(dataset2, dataset1)

    def export_as_csv(self, filename: 'Path', mnt_dir: 'str', verbose: 'bool' = False, threads: int = 1):
        df = pd.DataFrame(columns=["ds", "sg", "n_img", "t_id", "node1", "edge", "node2"])
        with Pool(processes=threads) as pool:
            results = []
            for key, value in tqdm(self._clusters.items(), disable=not verbose):
            # for (param1, param2) in param_list:
                results.append(pool.apply_async(self.graph_to_df, (key, value,mnt_dir)))
            # once submitted, all are running
            for result in tqdm(results):
                return_value = result.get()  # this will wait until the result is ready
                df = df.append(return_value)
        df.to_csv(filename, index=False)

    def graph_to_df(self, key, value, mnt_dir):
        df = pd.DataFrame(columns=["ds", "sg", "n_img", "t_id", "node1", "edge", "node2"])
        sg = load_sg(self._sg_files[key])
        node_list = sg.nodes()
        tuple_id = 0
        n_img = len(value)
        # Traverse SG nodes
        for node_a_id, node_a in enumerate(node_list):
            # Ignore base level nodes
            if node_a.name in ["Root Road", "Left Lane", "Middle Lane", "Right Lane"]:
                continue
            # Get successors of all other nodes
            else:
                for succesor in rx.bfs_successors(sg, node_a_id):
                    # Only interested in successors within depth 1
                    if succesor[0].name == node_a.name:
                        # Iterate over nodes connected to node_a
                        for node_b in sorted(list(set(succesor[1])), key=lambda x: x.name):
                            edges = sg.get_all_edge_data(node_a_id, node_list.index(node_b))
                            # Iterate over edges between node_a and node_b
                            for edge in edges:
                                img_idx = os.path.relpath(key, mnt_dir)
                                row = [
                                    img_idx.split("/")[0],
                                    img_idx,
                                    n_img,
                                    tuple_id,
                                    node_a.name,
                                    edge["label"],
                                    node_b.name
                                ]
                                tuple_id += 1
                                row_id = len(df)
                                df.loc[row_id] = row
        return df