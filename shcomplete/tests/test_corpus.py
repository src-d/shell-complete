import argparse
import tempfile
import unittest

from shcomplete.corpus import write_corpus


class CorpusTests(unittest.TestCase):

    def test_write_vocab(self):
        with tempfile.NamedTemporaryFile(prefix="shcomplete-test-corpus", suffix=".txt") as tmpf:
            args = argparse.Namespace(data_directory="shcomplete/tests/data/histories",
                                      output=tmpf.name)
            write_corpus(args)
            corpus = tmpf.read().decode("utf-8")
        self.assertIsInstance(corpus, str)
        self.assertEqual(len(corpus.split("\n\n")), 4)


if __name__ == '__main__':
    unittest.main()
