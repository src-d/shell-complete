import logging
import os
from clint.textui import progress

import pygtrie


def build_trie(lines, max_len=8):
    """
    Return a trie built with the prefixes encountered in filename
    """
    tf_trie = pygtrie.StringTrie(separator=" ")
    for line in lines:
        line = [l for l in line.rstrip().replace("\t", " ").split(" ") if l][:max_len]
        if not line:
            continue
        key = " ".join(line)
        if key not in tf_trie:
            tf_trie[key] = 0
        node = tf_trie._root
        for token in line:
            node = node.children[token]
            try:
                node.value += 1
            except TypeError:
                node.value = 1
    for prefix in tf_trie.keys():
        tf_trie[prefix] /= len(lines)
    return tf_trie


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
            with open(path_to_file, "r") as f:
                lines = f.readlines()
            tf_trie = build_trie(lines, max_len)
            tries.append((path_to_file, tf_trie))
    return tries


def get_df_trie(tries):
    """
    Return a trie where the prefix's value is its document frequency among the tries.
    """
    df_trie = pygtrie.StringTrie(separator=" ")
    for _, trie in progress.bar(tries, expected_size=len(tries)):
        for prefix in trie.keys():
            if prefix not in df_trie:
                df_trie[prefix] = 0
            df_trie[prefix] += 1
    for prefix in df_trie.keys():
        df_trie[prefix] = (df_trie[prefix] - 1) / len(tries)
    return df_trie


def sum_tfdf_tries(tries, df_trie, _log):
    """
    Compute the TF-DF scores of each prefix in each trie.
    Sum these scores in one big trie that is returned.
    """
    tfdf_trie = pygtrie.StringTrie(separator=" ")
    for path_to_file, trie in progress.bar(tries, expected_size=len(tries)):
        stack = [("", trie._root, df_trie._root)]
        while stack:
            path, local_node, df_node = stack.pop(0)
            for child_key, child_node in local_node.children.items():
                stack.append((((path + " ") if path else "") + child_key,
                             child_node, df_node.children[child_key]))
            try:
                tfdf = local_node.value * df_node.value
                tfdf_trie[path] = tfdf_trie.get(path, 0) + tfdf
            except TypeError:
                continue
    return tfdf_trie


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
    tfdf_trie = sum_tfdf_tries(tries, df_trie, _log)
    pruned_trie = prune(tfdf_trie, threshold=args.threshold)

    vocab = set()
    for prefix in pruned_trie.keys():
        vocab.add(prefix)

    vocab = list(vocab)
    with open(args.output, "w") as fout:
        fout.write("UNK")
        for prefix in vocab:
            fout.write("\n" + prefix)
