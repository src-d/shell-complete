import argparse
import logging
import tempfile
import unittest

from shcomplete.repos import to_raw_urls, fetch_repos


class RawUrlsTest(unittest.TestCase):

    def setUp(self):
        self.base = "https://raw.githubusercontent.com"
        self.url1 = "https://github.com/src-d/shell-complete/blob/master/README.md"
        self.url2 = "https://github.com/src-d/ast2vec/blob/master/ast2vec/tests/test_prox.py"
        self.urls = {self.url1, self.url2}

    def test_remove_blob(self):
        for url in to_raw_urls(self.urls, self.base):
            self.assertFalse("blob" in url)

    def test_url_prefix(self):
        for url in to_raw_urls(self.urls, self.base):
            self.assertTrue(url.startswith(self.base))


class FetchReposTests(unittest.TestCase):

    def test_fetch(self):
        with tempfile.NamedTemporaryFile(prefix="shcomplete-test-repos", suffix=".txt") as tmpf:
            args = argparse.Namespace(token=None,
                                      timeout=100, per_page=100, nb_search=1, output=tmpf.name)
            fetch_repos(args, testing=True)
            repos = tmpf.read().decode("utf-8")
        self.assertIsInstance(repos, str)
        self.assertTrue(len(repos.split("\n")) >= 1)


if __name__ == '__main__':
    unittest.main()
