import logging
import random
import sys
from configparser import SafeConfigParser

from keras.callbacks import Callback
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np

from shcomplete.tfdf import build_trie


def get_config(config_file):
    """
    Return a dict where is store the parameters of the model from the configuration file.
    """
    parser = SafeConfigParser()
    parser.read(config_file)
    params_ints = [(key, int(value)) for key, value in parser.items("ints")]
    params_floats = [(key, float(value)) for key, value in parser.items("floats")]
    return dict(params_ints + params_floats)


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


class vocabulary(object):

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

    def decode(self, X):
        """
        Decode the numpy array X and return the corresponding string.
        """
        X = X.argmax(axis=-1)
        sequence = "\n".join(self.index2command[i] for i in X)
        return sequence


def vectorize(sequences, next_commands, vocab, path_to_vocab):
    """
    Vectorize the data in mumpy arrays.
    """
    X = np.zeros((len(sequences), gConfig["seq_len"], vocab.size), dtype=np.bool)
    y = np.zeros((len(sequences), vocab.size), dtype=np.bool)
    for i, sequence in enumerate(sequences):
        for j, command in enumerate(sequence):
            try:
                vocab_trie = vocab.trie(path_to_vocab)
                prefix = vocab_trie.longest_prefix(command)[0]
                X[i, j, vocab.command2index[prefix]] = 1
            except KeyError:
                X[i, j, vocab.command2index["UNK"]] = 1
        y[i, vocab.command2index[next_commands[i]]] = 1
    return X, y


def generator(path_to_corpus, file_delimiter, path_to_vocab):
    """
    Return a random batch of data to feed fit_generator.
    The ouput is a tuple (sequences, outputs)
    """
    vocab = vocabulary(path_to_vocab)
    sequences = []
    next_commands = []
    while True:
        with open(path_to_corpus, "r") as f:
            histories = f.read().split(file_delimiter)
            for history in histories:
                history = history.split("\n")
                for _ in range(gConfig["batch_size"]):
                    ind = random.randint(0, len((history[:-gConfig["seq_len"]])) - 1)
                    sequence = [cmd.rstrip() for cmd in history[ind:ind + gConfig["seq_len"]]]
                    sequences.append(sequence)
                    vocab_trie = vocab.trie(path_to_vocab)
                    next_prefix = vocab_trie.longest_prefix(history[ind + gConfig["seq_len"]])[0]
                    if next_prefix is not None and next_prefix:
                        next_commands.append(next_prefix)
                    else:
                        next_commands.append("UNK")

                    if len(sequences) == gConfig["batch_size"]:
                        X, y = vectorize(sequences, next_commands, vocab, path_to_vocab)
                        yield X, y
                        sequences = []
                        next_commands = []


def generate_model(vocab):
    """
    Generate the model.
    """
    model = Sequential()
    for layer_id in range(gConfig["input_layers"]):
        model.add(LSTM(gConfig["hidden_size"], input_shape=(gConfig["seq_len"], vocab.size),
                       return_sequences=layer_id + 1 < gConfig["input_layers"]))
        model.add(Dropout(gConfig["amount_of_dropout"]))

    model.add(Dense(vocab.size))
    model.add(Activation("softmax"))
    optimizer = RMSprop(lr=gConfig["learning_rate"])
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def sample_prediction(model, X, y, vocab):
    """
    Select a sequence of shell commands and print the prediction of the model at current training.
    """
    index = random.randint(0, len(X)-1)
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


class OnEpochEndCallback(Callback):

    def on_epoch_end(self, epoch, logs=None):
        """
        Apply the prediction of the model to a batch of data at the end of each epoch.
        """
        vocab = vocabulary("vocab_0.01.txt")
        X_batch, y_batch = next(generator("corpus3.txt", "FILE_SEP\n", "vocab_0.01.txt"))
        sample_prediction(self.model, X_batch, y_batch, vocab)


ON_EPOCH_END_CALLBACK = OnEpochEndCallback()


def iterate(model, corpus, file_delimiter, path_to_vocab):
    """
    Iterative training of the model.
    """
    model.fit_generator(generator(corpus, file_delimiter, path_to_vocab),
                        samples_per_epoch=gConfig["steps_per_epoch"],
                        nb_epoch=gConfig["nb_epoch"],
                        callbacks=[ON_EPOCH_END_CALLBACK, ],
                        validation_data=None)


def train(args, log_level=logging.INFO):
    """
    Train the model and show the progress of the prediction at each epoch.
    """
    _log = logging.getLogger("training")
    _log.setLevel(log_level)

    global gConfig
    gConfig = get_config(args.config_file)
    vocab = vocabulary(args.vocabulary)
    model = generate_model(vocab)
    iterate(model, args.corpus, args.file_delimiter, args.vocabulary)
