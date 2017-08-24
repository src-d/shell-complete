import unittest

from shcomplete.repos import to_raw_urls


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


if __name__ == '__main__':
    unittest.main()
