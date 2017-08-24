import logging
import os
import re
from math import log
from clint.textui import progress

import pygtrie
import numpy


def get_tries(args, _log):
    """
    Return a list of tries, one for each file where the keys are the command lines.
    """
    tries = []
    max_len = args.max_length
    for root, _, files in os.walk(args.data_directory):
        for file_name in files:
            path_to_file = os.path.join(root, file_name)
            _log.info("parsing %s", path_to_file)
            trie = pygtrie.StringTrie(separator=" ")
            tries.append((path_to_file, trie))
            with open(path_to_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = [l for l in line.rstrip().replace("\t", " ").split(" ") if l][:max_len]
                    key = " ".join(line)
                    if key not in trie:
                        trie[key] = 0
                    node = trie._root
                    for token in line:
                        node = node.children[token]
                        try:
                            node.value += 1
                        except TypeError:
                            node.value = 1
    return tries


def get_df_trie(tries):
    """
    Return a trie where the prefix's value is its document frequency among the tries.
    """
    df_trie = pygtrie.StringTrie(separator=" ")
    for _, trie in progress.bar(tries, expected_size=len(tries)):
        for prefix, _ in trie.iteritems():
            if prefix not in df_trie:
                df_trie[prefix] = 0
            df_trie[prefix] += 1
    return df_trie


def sum_tfdf_tries(tries, df_trie, _log):
    """
    Compute the TF-DF scores of each prefix in each trie.
    Sum these scores in one big trie that is returned.
    """
    big_trie = pygtrie.StringTrie(separator=" ")
    for path_to_file, trie in progress.bar(tries, expected_size=len(tries)):
        _log.info("\n%s %d\n", path_to_file, len(trie))
        stack = [("", trie._root, df_trie._root)]
        while stack:
            path, local_node, df_node = stack.pop(0)
            for child_key, child_node in local_node.children.items():
                stack.append((((path + " ") if path else "") + child_key,
                             child_node, df_node.children[child_key]))
                try:
                    tfdf = (local_node.value / len(trie)) * ((df_node.value - 1) / len(tries))
                    print(path)
                    big_trie[path] = big_trie.get(path, 0) + tfdf
                except TypeError:
                    continue
    return big_trie


def prune(trie, threshold):
    """
    Prune the branches that are connected to the trie by a prefix value lower than threshold.
    """
    queue = [("", trie._root)]
    while queue:
        path, local_node = queue.pop(0)
        branch2prune = []
        for child_key, child_node in local_node.children.items():
            key = path + " " + child_key
            prefix = key[1:]
            if child_node.value < threshold:
                branch2prune.append(prefix)
            else:
                queue.append((path + " " + child_key, child_node))
        for prefix in branch2prune:
            del trie[prefix:]
    return trie


def filter_prediction_set(args, log_level=logging.INFO):
    """
    Return the set of prefixes among which we want to predict the next command.
    The separator must be a string that could not be encountered in command lines of any history.
    """
    _log = logging.getLogger("tfdf")
    _log.setLevel(log_level)

    tries = get_tries(args, _log)
    df_trie = get_df_trie(tries)
    big_tfdf_trie = sum_tfdf_tries(tries, df_trie, _log)
    pruned_trie = prune(big_tfdf_trie, threshold=args.threshold)

    vocab = set()
    for prefix in pruned_trie.keys():
        vocab.add(prefix)

    vocab = list(vocab)
    with open(args.output, "w") as fout:
        for prefix in vocab:
            fout.write(prefix + "\n")
