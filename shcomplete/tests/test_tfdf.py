import argparse
import logging
import unittest

from shcomplete.tfdf import get_tries, get_df_trie, sum_tfdf_tries, prune


def get_values(tries):
    list_items = []
    for _, trie in tries:
        for prefix, value in trie.iteritems():
            list_items.append((prefix, value))
    return list_items


class TestTries(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(max_length=8, threshold=0.01,
                                       data_directory="shcomplete/tests/data")
        self._log = logging.getLogger("test")
        self._log.setLevel(logging.WARNING)
        self.tries = get_tries(self.args, self._log)
        self.list_items = get_values(self.tries)
        self.df_trie = get_df_trie(self.tries)
        self.tfdf_trie = sum_tfdf_tries(self.tries, self.df_trie, self._log)

    def test_list(self):
        self.assertEqual(len(self.tries), 3)

    def test_in_tries(self):
        for _, trie in self.tries:
            self.assertIn("git", trie)
            self.assertNotIn("git ", trie)
            self.assertIn("git push", trie)
            self.assertIn("git status", trie)

    def test_max_length(self):
        for prefix, _ in self.list_items:
            prefix = prefix.split(" ")
            self.assertTrue(len(prefix) <= self.args.max_length)

    def test_tf_values(self):
        for _, value in self.list_items:
            self.assertTrue(value > 0)
            self.assertTrue(value <= 1)

    def test_df_trie(self):

        for prefix, value in self.list_items:
            self.assertIn(prefix, self.df_trie)
            self.assertTrue(value <= 1)

    def test_sum_tfdf(self):
        for prefix, value in self.tfdf_trie.iteritems():
            self.assertTrue(value >= 0)
            self.assertTrue(prefix, self.df_trie)
            self.assertEqual(len(self.tfdf_trie), len(self.df_trie))

    def test_prune(self):
        pruned_trie = prune(self.tfdf_trie, self.args.threshold)
        for prefix, value in pruned_trie.iteritems():
            self.assertTrue(prefix, self.tfdf_trie)
            self.assertTrue(value >= self.args.threshold)
            self.assertTrue(len(pruned_trie) <= len(self.tfdf_trie))


if __name__ == '__main__':
    unittest.main()
