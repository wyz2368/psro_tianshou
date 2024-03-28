import os
import pickle

import numpy as np


def isExist(path):
    """
    Check if a path exists.
    :param path: path to check.
    :return: bool
    """
    return os.path.exists(path)

def save_pkl(obj,path):
    """
    Pickle a object to path.
    :param obj: object to be pickled.
    :param path: path to save the object
    """
    with open(path,'wb') as f:
        pickle.dump(obj,f)

def load_pkl(path):
    """
    Load a pickled object from path
    :param path: path to the pickled object.
    :return: object
    """
    if not isExist(path):
        raise ValueError(path + " does not exist.")
    with open(path,'rb') as f:
        result = pickle.load(f)
    return result


def find_first_greater(nums, target):
    """
    Finds the first position in a sorted list where the number is greater than a given number.

    Args:
    - nums (list): Sorted list of numbers.
    - target (int or float): The number to compare with.

    Returns:
    - int: The index of the first position where the number is greater than the given number.
    """
    left = 0
    right = len(nums) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] > target:
            result = mid
            right = mid - 1
        else:
            left = mid + 1

    return result

def organize_nodes_in_layers(nodes_per_layer, subset_nodes):
    organized_layers = []
    last_idx = 0
    subset_nodes = list(sorted(subset_nodes))
    subset_nodes.append(np.inf)
    for nd in np.cumsum(nodes_per_layer):
        idx = find_first_greater(subset_nodes, nd-1)
        if idx == -1:
            organized_layers.append([])
        else:
            organized_layers.append(list(subset_nodes[last_idx: idx]))
        last_idx = idx

    return organized_layers

def find_all_paths(organized_layers, adj_m):
    def backtrack(curr_path, idx):
        if idx == len(organized_layers):
            paths.append(curr_path[:])
            return
        for num in organized_layers[idx]:
            if len(curr_path) == 0 or num in adj_m[curr_path[-1]]:
                curr_path.append(num)
                backtrack(curr_path, idx + 1)
                curr_path.pop()

    paths = []
    backtrack([], 0)
    return paths


def compute_linkages(nodes_per_layer, subset_nodes, adj_m):
    organized_layers = organize_nodes_in_layers(nodes_per_layer, subset_nodes)
    paths = find_all_paths(organized_layers, adj_m)
    return paths
























