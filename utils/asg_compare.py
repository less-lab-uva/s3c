import ctypes
import multiprocessing
from functools import lru_cache
from multiprocessing import Process

import rustworkx as rx


def remove_ids(label):
    """
    Remove the id number from the label

    Assumes that if an id is present, it is of the form _number, e.g. car_2
    """
    if label is None:
        return None
    if not (hasattr(label, 'name') and hasattr(label, 'label')):
        if isinstance(label, dict) and 'label' in label:
            return label['label']
        if hasattr(label, 'name'):
            under_index = label.name.rfind('_')
            if under_index == -1:
                return label.name
            else:
                return label.name[:under_index]
        return label
    under_index = label.name.rfind('_')
    return '%s:%s' % (label.label, label.name[:under_index] if under_index > 0 else label.name)


@lru_cache
def get_hierarchy_check(force_consistent_ids=False):
    def hierarchy_check(label1, label2):
        # If they have the same label, they are equal
        if not force_consistent_ids:
            label1 = remove_ids(label1)
            label2 = remove_ids(label2)
        if label1 == label2:
            return True
    return hierarchy_check


def compare_asgs(asg1, asg2):
    return rx.digraph_is_isomorphic(asg1, asg2, id_order=False,
                                    node_matcher=get_hierarchy_check(),
                                    edge_matcher=get_hierarchy_check())


def dict_to_string(d: dict):
    return str({key: d[key] for key in sorted(d.keys())})


def get_class_counts(asg):
    """
    Returns a tuple of dictionaries (node_class_counts, edge_class_counts) with the class counts.

    If two graphs are isomorphic, then they must have the same number of every node and edge label (here called "class")
    This is used as a lightweight check as part of maybe_isomorphic to avoid running the more expensive algorithm.
    """
    node_class_counts = {}
    for node_index in asg.node_indices():
        label = remove_ids(asg[node_index])
        if label not in node_class_counts:
            node_class_counts[label] = 0
        node_class_counts[label] += 1
    edge_class_counts = {}
    for edge_index, (node1, node2, edge_data) in asg.edge_index_map().items():
        label = remove_ids(edge_data)
        if label not in edge_class_counts:
            edge_class_counts[label] = 0
        edge_class_counts[label] += 1
    return dict_to_string(node_class_counts), dict_to_string(edge_class_counts)


def maybe_isomorphic(asg1, asg2):
    """Check metrics which are necessary but not sufficient for isomorphism and cheap to compute"""
    # if the graphs don't have the same size then they can't be isomorphic
    if asg1.num_nodes() != asg2.num_nodes() or asg1.num_edges() != asg2.num_edges():
        return False
    else:
        # if they don't have the same number of nodes and edges with equivalent labels, they can't be isomorphic
        asg1_class_counts = get_class_counts(asg1)
        asg2_class_counts = get_class_counts(asg2)
        if asg1_class_counts != asg2_class_counts:
            return False
    return True


def __is_isomorphic(asg1, asg2, value: multiprocessing.Value = None):
    """Checks isomorphism directly. For improved performance, check maybe_isomorphic first"""
    result = compare_asgs(asg1, asg2)
    if value is not None:
        value.value = 1 if result else -1
    return result


def is_isomorphic(asg1, asg2, timeout=-1, check_preconditions=True):
    """
     Strict comparison assuming perfect equality. Can be run with timeout.
     If check_preconditions, checks maybe_isomorphic first. That time is excluded from the timeout.

     If timeout <=0, run without time limit. If timeout>0, run for at mose timeout seconds.
     If no result is found within that time, None is returned.
    """
    if check_preconditions and not maybe_isomorphic(asg1, asg2):
        return False
    if timeout <= 0:
        is_iso = __is_isomorphic(asg1, asg2)
    else:
        return_value = multiprocessing.Value(ctypes.c_byte, 0)
        p = Process(target=__is_isomorphic, args=(asg1, asg2, return_value))
        p.start()
        p.join(timeout)
        if p.is_alive() or return_value.value == 0:
            p.terminate()
            p.join()
            is_iso = None
        else:
            is_iso = return_value.value > 0
    return is_iso
