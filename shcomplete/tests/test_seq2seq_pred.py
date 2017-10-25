import os
import unittest

import numpy as np

from shcomplete.seq2seq_prediction import get_config, Vocabulary, generate_model
from shcomplete.seq2seq_prediction import generator, sample_prediction, OnEpochEndCallback


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.gConfig = get_config("shcomplete/seq2seq.ini")

    def test_type(self):
        self.assertIsInstance(self.gConfig, dict)
        self.assertIsInstance(self.gConfig["nb_epoch"], int)
        self.assertIsInstance(self.gConfig["learning_rate"], float)

    def test_in_config(self):
        self.assertIn("seq_len", self.gConfig)
        self.assertIn("batch_size", self.gConfig)
        self.assertIn("hidden_layers", self.gConfig)

    def test_upperbound(self):
        self.assertTrue(self.gConfig["amount_of_dropout"] < 1)
        self.assertTrue(self.gConfig["learning_rate"] < 1)


class DataGenerator(unittest.TestCase):
    def setUp(self):
        self.gConfig = get_config("shcomplete/seq2seq.ini")
        self.path_to_vocab = "shcomplete/tests/data/vocab_0.01.txt"
        self.vc = Vocabulary(self.path_to_vocab)
        self.trie = self.vc.trie(self.path_to_vocab)
        self.path_to_corpus = "shcomplete/tests/data/corpus.txt"
        self.file_delimiter = "FILE DELIMITER\n"

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

    def test_decoder(self):
        X = np.zeros((2, self.vc.size))
        X[0, 127] = 1
        X[1, 71] = 1
        y = np.zeros((1, self.vc.size))
        y[0, 59] = 0.7
        self.assertEqual(self.vc.decode(X), "\n".join(["git commit -a", "git push origin master"]))
        self.assertEqual(self.vc.decode(y), "git checkout")

    def test_generator(self):
        X, y = next(generator(self.path_to_corpus, self.file_delimiter,
                              self.path_to_vocab, self.gConfig))
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape,
                         (self.gConfig["batch_size"], self.gConfig["seq_len"], self.vc.size))
        self.assertEqual(y.shape, (self.gConfig["batch_size"], self.vc.size))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.gConfig = get_config("shcomplete/seq2seq.ini")
        self.path_to_vocab = "shcomplete/tests/data/vocab_0.01.txt"
        self.vc = Vocabulary(self.path_to_vocab)
        self.path_to_corpus = "shcomplete/tests/data/corpus.txt"
        self.file_delimiter = "FILE DELIMITER\n"
        self.model = generate_model(self.path_to_vocab, self.gConfig)
        self.models_directory = "shcomplete/tests/data"

    def test_generate_model(self):
        self.assertTrue(self.model.built)
        self.assertIn("sequential", self.model.name)
        self.assertTrue(self.model.trainable)
        self.assertIsInstance(self.model.get_weights()[0], np.ndarray)
        self.assertEqual(self.model.get_weights()[0].shape,
                         (self.vc.size, self.gConfig["hidden_layers"]))
        self.assertTrue(self.model.model.uses_learning_phase)

    def test_callback(self):
        ON_EPOCH_END_CALLBACK = OnEpochEndCallback(self.path_to_vocab, self.file_delimiter,
                                                   self.path_to_corpus, self.models_directory,
                                                   self.gConfig)
        self.model.fit_generator(generator(self.path_to_corpus, self.file_delimiter,
                                           self.path_to_vocab, self.gConfig),
                                 samples_per_epoch=2,
                                 nb_epoch=1,
                                 callbacks=[ON_EPOCH_END_CALLBACK, ],
                                 validation_data=None)
        self.assertTrue(os.path.isfile("shcomplete/tests/data/keras_pred_e0.h5"))

    def test_sample_prediction(self):
        X, y = next(generator(self.path_to_corpus, self.file_delimiter,
                              self.path_to_vocab, self.gConfig))
        seq, next_cmd_pred, next_cmd_true = sample_prediction(self.model, X, y, self.vc)
        self.assertIsInstance(next_cmd_pred, str)
        self.assertIn(next_cmd_pred, self.vc.trie(self.path_to_vocab))
        self.assertEqual(len(seq.split("\n")), self.gConfig["seq_len"])
        self.assertEqual(len(next_cmd_true.split("\n")), 1)


if __name__ == '__main__':
    unittest.main()
