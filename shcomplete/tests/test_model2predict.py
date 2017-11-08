import argparse
import logging
import os
import unittest

import numpy as np

from shcomplete.model2predict import Vocabulary, generate_model, train_predict
from shcomplete.model2predict import generator, sample_prediction


class DataGenerator(unittest.TestCase):
    def setUp(self):
        self.path_to_vocab = "shcomplete/tests/data/vocab_0.01.txt"
        self.vc = Vocabulary(self.path_to_vocab)
        self.trie = self.vc.trie(self.path_to_vocab)
        self.path_to_corpus = "shcomplete/tests/data/corpus.txt"

    def test_voc_size(self):
        self.assertEqual(self.vc.size, len(self.vc.vocab))

    def test_trie(self):
        self.assertIn("git", self.trie)
        self.assertNotIn("git ", self.trie)
        self.assertIn("git push", self.trie)
        self.assertIn("git status", self.trie)
        for _, value in self.trie.iteritems():
            self.assertTrue(value > 0)
            self.assertTrue(value <= 1)

    def test_encoder(self):
        y = self.vc.encode(["git merge develop", "specific_command"], self.trie, 5, target=True)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y[0], self.vc.command2index["git merge"])
        self.assertEqual(y[1], self.vc.command2index["UNK"])

    def test_decoder(self):
        X = np.zeros(2)
        X[0] = self.vc.command2index["git commit -a"]
        X[1] = self.vc.command2index["git push origin master"]
        y = np.zeros((1, self.vc.size))
        y[0, self.vc.command2index["git checkout"]] = 0.7
        self.assertEqual(self.vc.decode(X), "\n".join(["git commit -a", "git push origin master"]))
        self.assertEqual(self.vc.decode(y, reduction=True), "git checkout")

    def test_generator(self):
        X, y = next(generator(self.path_to_corpus, self.path_to_vocab,
                              batch_size=32, seq_len=10))
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape, (32, 10))
        self.assertEqual(y.shape, (32, 1, 1))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger("test")
        self.log.setLevel(logging.INFO)
        self.path_to_vocab = "shcomplete/tests/data/vocab_0.01.txt"
        self.vc = Vocabulary(self.path_to_vocab)
        self.path_to_corpus = "shcomplete/tests/data/corpus.txt"
        self.model_directory = "shcomplete/tests/data"
        self.args = argparse.Namespace(vocabulary=self.path_to_vocab, corpus=self.path_to_corpus,
                                       model_directory=self.model_directory, seq_len=10,
                                       input_layers=1, hidden_layers=2, output_layers=1,
                                       dropout=0.2, batch_size=32, nb_epochs=2, steps_per_epoch=64,
                                       from_model=None, checkpoint=2)
        self.model = generate_model(self.args)

    def test_generate_model(self):
        self.assertTrue(self.model.built)
        self.assertIn("sequential", self.model.name)
        self.assertTrue(self.model.trainable)
        self.assertIsInstance(self.model.get_weights()[0], np.ndarray)
        self.assertEqual(self.model.get_weights()[0].shape, (self.vc.size, self.vc.size))
        self.assertTrue(self.model.model.uses_learning_phase)

    def test_sample_prediction(self):
        X, y = next(generator(self.path_to_corpus, self.path_to_vocab,
                              batch_size=32, seq_len=10))
        seq, next_cmd_pred, next_cmd_true = sample_prediction(self.model, X, y, self.vc)
        self.assertIsInstance(next_cmd_pred, str)
        self.assertIn(next_cmd_pred, self.vc.trie(self.path_to_vocab))
        self.assertEqual(len(seq.split("\n")), 10)
        self.assertEqual(len(next_cmd_true.split("\n")), 1)

    def test_train_correct(self):
        train_predict(self.args)
        self.assertTrue(os.path.isfile("shcomplete/tests/data/keras_pred_e0.h5"))
        self.args.from_model = "shcomplete/tests/data/keras_pred_e0.h5"
        self.args.nb_epochs = 3
        train_predict(self.args)
        self.assertTrue(os.path.isfile("shcomplete/tests/data/keras_pred_e0.h5"))


if __name__ == '__main__':
    unittest.main()
