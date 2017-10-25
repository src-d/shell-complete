import os
import unittest

import numpy as np

from shcomplete.seq2seq_prediction import get_config
from shcomplete.seq2seq_correction import Seq2seq, generate_model, get_chars
from shcomplete.seq2seq_correction import generator, sample_prediction, OnEpochEndCallback


class DataGenerator(unittest.TestCase):
    def setUp(self):
        self.gConfig = get_config("shcomplete/seq2seq.ini")
        self.path_to_vocab = "shcomplete/tests/data/vocab_0.01.txt"
        self.chars = get_chars(self.path_to_vocab)
        self.seq2seq = Seq2seq(self.chars)
        self.path_to_corpus = "shcomplete/tests/data/corpus.txt"
        self.file_delimiter = "FILE DELIMITER\n"

    def test_encoder(self):
        X = self.seq2seq.encode(["git merge develop", "git checkout"], max_cmd_len=20)
        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(X.shape, (2, 20, len(self.chars)))

    def test_decoder(self):
        print(self.seq2seq.char_indices)
        X = np.zeros((20, len(self.chars)))
        X[0, self.seq2seq.char_indices["g"]] = 1
        X[1, self.seq2seq.char_indices["i"]] = 1
        X[2, self.seq2seq.char_indices["t"]] = 1
        X[3, self.seq2seq.char_indices["k"]] = 1
        self.assertEqual(self.seq2seq.decode(X, 20), "gitk")

    def test_generator(self):
        X, y = next(generator(self.chars, self.path_to_corpus, self.file_delimiter,
                              self.path_to_vocab, self.gConfig))
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape, (self.gConfig["batch_size"],
                                   self.gConfig["max_cmd_len"],
                                   len(self.chars)))
        self.assertEqual(y.shape, (self.gConfig["batch_size"],
                                   self.gConfig["max_cmd_len"],
                                   len(self.chars)))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.gConfig = get_config("shcomplete/seq2seq.ini")
        self.path_to_vocab = "shcomplete/tests/data/vocab_0.01.txt"
        self.chars = get_chars(self.path_to_vocab)
        self.path_to_corpus = "shcomplete/tests/data/corpus.txt"
        self.file_delimiter = "FILE DELIMITER\n"
        self.model = generate_model(self.chars, self.gConfig)
        self.models_directory = "shcomplete/tests/data"

    def test_generate_model(self):
        self.assertTrue(self.model.built)
        self.assertIn("sequential", self.model.name)
        self.assertTrue(self.model.trainable)
        self.assertIsInstance(self.model.get_weights()[0], np.ndarray)
        self.assertEqual(self.model.get_weights()[0].shape,
                         (len(self.chars), self.gConfig["hidden_layers"]))
        self.assertTrue(self.model.model.uses_learning_phase)

    def test_callback(self):
        ON_EPOCH_END_CALLBACK = OnEpochEndCallback(self.path_to_vocab, self.file_delimiter,
                                                   self.path_to_corpus, self.models_directory,
                                                   self.gConfig)
        self.model.fit_generator(generator(self.chars, self.path_to_corpus, self.file_delimiter,
                                           self.path_to_vocab, self.gConfig),
                                 samples_per_epoch=2,
                                 nb_epoch=1,
                                 callbacks=[ON_EPOCH_END_CALLBACK, ],
                                 validation_data=None)
        self.assertTrue(os.path.isfile("shcomplete/tests/data/keras_spell_e0.h5"))

    def test_sample_prediction(self):
        X, y = next(generator(self.chars, self.path_to_corpus, self.file_delimiter,
                              self.path_to_vocab, self.gConfig))
        predicted_cmd = sample_prediction(self.model, self.chars, X, y,
                                          self.gConfig["max_cmd_len"],
                                          self.gConfig["inverted"])
        self.assertIsInstance(predicted_cmd, str)
        self.assertTrue(len(predicted_cmd) <= self.gConfig["max_cmd_len"])


if __name__ == '__main__':
    unittest.main()
