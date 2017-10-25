import logging
import os
import sys
from configparser import ConfigParser

from keras.callbacks import Callback
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.models import Sequential, load_model
import numpy as np
from numpy.random import randint

from shcomplete.tfdf import build_trie


def get_config(config_file):
    """
    Return a dict where is store the parameters of the model from the configuration file.
    """
    parser = ConfigParser()
    parser.read(config_file)
    params_ints = [(key, int(value)) for key, value in parser.items("ints")]
    params_floats = [(key, float(value)) for key, value in parser.items("floats")]
    params_strings = [(key, str(value)) for key, value in parser.items("strings")]
    return dict(params_ints + params_floats + params_strings)


def get_vocabulary(path_to_vocab):
    """
    Return a list of prefixes making the vocabulary.
    """
    vocab = []
    with open(path_to_vocab, "r") as f:
        for line in f:
            line = line.rstrip()
            vocab.append(line)
    return vocab


class Vocabulary(object):

    def __init__(self, path_to_vocab):
        self.vocab = get_vocabulary(path_to_vocab)
        self.command2index = dict((c, i) for i, c in enumerate(self.vocab))
        self.index2command = dict((i, c) for i, c in enumerate(self.vocab))

    @property
    def size(self):
        """
        Return the number of prefixes in the vocabulary.
        """
        return len(self.vocab)

    def trie(self, path_to_vocab):
        """
        Return the vocabulary stored in a trie structure.
        """
        with open(path_to_vocab, "r") as f:
            content = f.read().split("\n")
        return build_trie(content)

    def encode(self, batch_of_data, vocab_trie, seq_len):
        """
        Encode sequences of commands into numpy arrays.
        """
        if len(batch_of_data[0]) == seq_len:
            X = np.zeros((len(batch_of_data), seq_len, self.size), dtype=np.bool)
            for i, sequence in enumerate(batch_of_data):
                for j, cmd in enumerate(sequence):
                    try:
                        prefix = vocab_trie.longest_prefix(cmd)[0]
                        X[i, j, self.command2index[prefix]] = 1
                    except KeyError:
                        X[i, j, self.command2index["UNK"]] = 1
            return X
        else:
            assert type(batch_of_data[0]) == str
            y = np.zeros((len(batch_of_data), self.size), dtype=np.bool)
            for i, sequence in enumerate(batch_of_data):
                try:
                    next_prefix = vocab_trie.longest_prefix(batch_of_data[i])[0]
                    y[i, self.command2index[next_prefix]] = 1
                except KeyError:
                    y[i, self.command2index["UNK"]] = 1
            return y

    def decode(self, X):
        """
        Decode the numpy array X and return the corresponding command.
        """
        X = X.argmax(axis=-1)
        sequence = "\n".join(self.index2command[i] for i in X)
        return sequence


def generator(path_to_corpus, file_delimiter, path_to_vocab, gConfig):
    """
    Return a random batch of data to feed fit_generator.
    The ouput is a tuple (sequences, outputs)
    """
    vocab = Vocabulary(path_to_vocab)
    vocab_trie = vocab.trie(path_to_vocab)
    sequences = []
    next_commands = []
    while True:
        with open(path_to_corpus, "r") as f:
            histories = f.read().split(file_delimiter)
            id = randint(0, len(histories)-2)
            history = histories[id].split("\n")
            for _ in range(gConfig["batch_size"]):
                ind = randint(0, len((history[:-gConfig["seq_len"]])) - 1)
                sequence = [cmd.rstrip() for cmd in history[ind:ind + gConfig["seq_len"]]]
                sequences.append(sequence)
                next_commands.append(history[ind + gConfig["seq_len"]])
            assert len(sequences) == gConfig["batch_size"]
            assert len(sequences) == len(next_commands)
            X = vocab.encode(sequences, vocab_trie, gConfig["seq_len"])
            y = vocab.encode(next_commands, vocab_trie, gConfig["seq_len"])
            sequences = []
            next_commands = []
            yield X, y


def generate_model(path_to_vocab, gConfig):
    """
    Generate the model.
    """
    vocab_size = Vocabulary(path_to_vocab).size
    model = Sequential()
    for layer_id in range(gConfig["input_layers"]):
        model.add(LSTM(gConfig["hidden_layers"], input_shape=(gConfig["seq_len"], vocab_size),
                       return_sequences=layer_id + 1 < gConfig["input_layers"]))
        model.add(Dropout(gConfig["amount_of_dropout"]))

    model.add(Dense(vocab_size))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def sample_prediction(model, X, y, vocab):
    """
    Select a sequence of shell commands and print the current prediction of the model at current.
    """
    index = randint(0, len(X)-1)
    rowX = X[np.array([index])]
    rowy = y[np.array([index])]

    sequence = vocab.decode(rowX[0])
    preds = model.predict(rowX, verbose=0)[0]
    preds = np.reshape(preds, (1, vocab.size))
    y = np.reshape(rowy[0], (1, vocab.size))
    next_command_pred = vocab.decode(preds)
    next_command_true = vocab.decode(y)

    print("----- Generating from seed -----")
    sys.stdout.write(sequence + "\n")

    print("----- Next command prediction -----")
    sys.stdout.write(next_command_pred + "\n")

    print("----- Next command True -----")
    sys.stdout.write(next_command_true + "\n")
    sys.stdout.flush()
    print()
    return sequence, next_command_pred, next_command_true


class OnEpochEndCallback(Callback):

    def __init__(self, path_to_vocab, file_delimiter, path_to_corpus, models_directory, gConfig):
        self.gConfig = gConfig
        self.vocab = path_to_vocab
        self.file_delimiter = file_delimiter
        self.corpus = path_to_corpus
        self.models_directory = models_directory

    def on_epoch_end(self, epoch, logs=None):
        """
        Apply the prediction of the model to a batch of data at the end of each epoch.
        """
        vocab = Vocabulary(self.vocab)
        X_batch, y_batch = next(generator(self.corpus, self.file_delimiter,
                                          self.vocab, self.gConfig))
        sample_prediction(self.model, X_batch, y_batch, vocab)
        path_to_model = os.path.join(self.models_directory, "keras_pred_e{}.h5".format(epoch))
        if epoch % 100 == 0:
            self.model.save(path_to_model)


def train_pred(args, log_level=logging.INFO):
    """
    Train the model and show the progress of the prediction at each epoch.
    """
    _log = logging.getLogger("training")
    _log.setLevel(log_level)

    if args.from_model:
        model = load_model(args.from_model)
    else:
        global gConfig
        gConfig = get_config(args.config_file)
        model = generate_model(args.vocabulary, gConfig)

    ON_EPOCH_END_CALLBACK = OnEpochEndCallback(args.vocabulary, args.file_delimiter,
                                               args.corpus, args.models_directory, gConfig)
    model.fit_generator(generator(args.corpus, args.file_delimiter, args.vocabulary, gConfig),
                        samples_per_epoch=gConfig["steps_per_epoch"],
                        nb_epoch=gConfig["nb_epoch"],
                        callbacks=[ON_EPOCH_END_CALLBACK, ],
                        validation_data=None)
