import argparse
import logging
import os
import unittest

from keras.layers import recurrent
import numpy as np

from shcomplete.model2correct import Seq2seq, generate_model, get_chars, train_correct
from shcomplete.model2correct import generator_misprints, dislpay_sample_correction


class DataGenerator(unittest.TestCase):
    def setUp(self):
        self.path_to_vocab = "shcomplete/tests/data/vocab_0.01.txt"
        self.chars = get_chars(self.path_to_vocab)
        self.seq2seq = Seq2seq(self.chars)
        self.path_to_corpus = "shcomplete/tests/data/corpus.txt"

    def test_encoder(self):
        X = self.seq2seq.encode(["git merge develop", "git checkout"])
        self.assertIsInstance(X, list)
        self.assertEqual(len(X), 2)
        self.assertEqual(len(X[0]), len("git merge develop"))

    def test_decoder(self):
        X = np.zeros(20)
        X[0] = self.seq2seq.char_indices["g"]
        X[1] = self.seq2seq.char_indices["i"]
        X[2] = self.seq2seq.char_indices["t"]
        X[3] = self.seq2seq.char_indices["k"]
        self.assertEqual(self.seq2seq.decode(X), "gitk")
        Z = np.zeros(20)
        Z[0] = self.seq2seq.char_indices["g"]
        Z[1] = self.seq2seq.char_indices["i"]
        Z[2] = self.seq2seq.char_indices["t"]
        Z[4] = self.seq2seq.char_indices["k"]
        self.assertEqual(self.seq2seq.decode(Z, reduction=False), "gitk")
        Z_null = np.zeros(20)
        self.assertEqual(self.seq2seq.decode(Z_null, reduction=False), "")

    def test_generator(self):
        self.args = argparse.Namespace(vocabulary=self.path_to_vocab, corpus=self.path_to_corpus,
                                       max_cmd_len=40, batch_size=32, level_noise=0.4)
        X, y = next(generator_misprints(self.args))
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape, (32, 40))
        self.assertEqual(y.shape, (32, 40, 1))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger("test")
        self.log.setLevel(logging.INFO)
        self.path_to_vocab = "shcomplete/tests/data/vocab_0.01.txt"
        self.chars = get_chars(self.path_to_vocab)
        self.path_to_corpus = "shcomplete/tests/data/corpus.txt"
        self.models_directory = "shcomplete/tests/data"
        self.args = argparse.Namespace(vocabulary=self.path_to_vocab, corpus=self.path_to_corpus,
                                       models_directory=self.models_directory, max_cmd_len=40,
                                       input_layers=1, hidden_layers=4, output_layers=1,
                                       dropout=0.2, batch_size=32, level_noise=0.4,
                                       nb_predictions=2, nb_epochs=1, steps_per_epoch=64,
                                       from_model=None, checkpoint=2, optimizer="adam",
                                       cell_type=recurrent.LSTM)
        self.model = generate_model(self.args, nb_features=len(self.chars)+1,
                                    input_length=self.args.max_cmd_len,
                                    nb_repeats=self.args.max_cmd_len)

    def test_generate_model(self):
        self.assertTrue(self.model.built)
        self.assertIn("sequential", self.model.name)
        self.assertTrue(self.model.trainable)
        self.assertIsInstance(self.model.get_weights()[0], np.ndarray)
        self.assertEqual(self.model.get_weights()[0].shape, (len(self.chars)+1, len(self.chars)+1))
        self.assertTrue(self.model.model.uses_learning_phase)

    def test_display_sample_correction(self):
        X, y = next(generator_misprints(self.args))
        correct_cmd = dislpay_sample_correction(self.args, self.model, X, y, nb_corrections=2)
        self.assertIsInstance(correct_cmd, str)
        self.assertTrue(len(correct_cmd) <= 40)

    def test_train_correct(self):
        train_correct(self.args)
        self.assertTrue(os.path.isfile("shcomplete/tests/data/keras_e0.h5"))
        self.args.from_model = "shcomplete/tests/data/keras_e0.h5"
        self.args.nb_epochs = 3
        train_correct(self.args)
        self.assertTrue(os.path.isfile("shcomplete/tests/data/keras_e2.h5"))


if __name__ == '__main__':
    unittest.main()
