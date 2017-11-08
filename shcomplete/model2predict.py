import logging
import os
import sys

from keras.callbacks import Callback
from keras.layers import Dense, Activation, recurrent, Embedding
from keras.layers import TimeDistributed, RepeatVector, Dropout
from keras.models import Sequential, load_model
import numpy as np
from numpy.random import randint

from shcomplete.tfdf import build_trie


def get_vocabulary(path_to_vocab):
    """
    Return a list of prefixes making the vocabulary.
    """
    vocab = []
    with open(path_to_vocab) as f:
        for line in f:
            line = line.rstrip()
            vocab.append(line)
    return vocab


class Vocabulary(object):
    def __init__(self, path_to_vocab):
        self.vocab = get_vocabulary(path_to_vocab)
        self.index2command = {}
        self.command2index = {}
        for i in range(len(self.vocab)):
            self.index2command[i] = self.vocab[i]
            self.command2index[self.vocab[i]] = i

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

    def encode(self, batch_of_data, vocab_trie, seq_len, target=False):
        """
        Encode sequences of commands into numpy arrays.
        """
        if target:
            y = np.zeros(len(batch_of_data))
            for i, command in enumerate(batch_of_data):
                try:
                    next_prefix = vocab_trie.longest_prefix(command)[0]
                    y[i] = self.command2index[next_prefix]
                except KeyError:
                    y[i] = self.command2index["UNK"]
            return y
        else:
            X = np.zeros((len(batch_of_data), seq_len))
            for i, sequence in enumerate(batch_of_data):
                for j, cmd in enumerate(sequence):
                    try:
                        prefix = vocab_trie.longest_prefix(cmd)[0]
                        X[i, j] = self.command2index[prefix]
                    except KeyError:
                        X[i, j] = self.command2index["UNK"]
            return X

    def decode(self, X, reduction=False):
        """
        Decode the numpy array X and return the corresponding command.
        """
        if reduction:
            X = X.argmax(axis=-1)
        sequence = "\n".join(self.index2command[i] for i in X)
        return sequence


def generator(path_to_corpus, path_to_vocab, batch_size, seq_len):
    """
    Return a random batch of data to feed fit_generator.
    The ouput is a tuple (sequences, outputs)
    """
    vocab = Vocabulary(path_to_vocab)
    vocab_trie = vocab.trie(path_to_vocab)
    with open(path_to_corpus) as f:
        histories = f.read().split("\n\n")
        while True:
            sequences = []
            next_commands = []
            while len(sequences) < batch_size:
                id_hist = randint(len(histories)-1)
                history = histories[id_hist].split("\n")
                id_cmd = randint(len(history[:-seq_len]))
                target = vocab_trie.longest_prefix(history[id_cmd + seq_len])[0]
                if target:
                    sequence = [cmd.rstrip() for cmd in history[id_cmd:id_cmd + seq_len]]
                    sequences.append(sequence)
                    next_commands.append(history[id_cmd + seq_len])
            X = vocab.encode(sequences, vocab_trie, seq_len)
            y = vocab.encode(next_commands, vocab_trie, seq_len, target=True)
            yield X, y[:, np.newaxis, np.newaxis]


def generate_model(args):
    """
    Generate the model.
    """
    nb_commands = Vocabulary(args.vocabulary).size
    emb_weights = np.eye(nb_commands)

    model = Sequential()
    model.add(Embedding(input_dim=nb_commands, output_dim=nb_commands, input_length=args.seq_len,
                        weights=[emb_weights], trainable=False))
    for layer_id in range(args.input_layers):
        model.add(recurrent.LSTM(args.hidden_layers,
                                 return_sequences=layer_id + 1 < args.input_layers))
        model.add(Dropout(args.dropout))

    model.add(RepeatVector(1))
    for _ in range(args.output_layers):
        model.add(recurrent.LSTM(args.hidden_layers, return_sequences=True))
        model.add(Dropout(args.dropout))

    model.add(TimeDistributed(Dense(nb_commands)))
    model.add(Activation("softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
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

    y = np.squeeze(rowy[0], axis=1)
    next_command_pred = vocab.decode(preds, reduction=True)
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
    def __init__(self, args, log):
        self.log = log
        self.vocabulary = args.vocabulary
        self.corpus = args.corpus
        self.batch_size = args.batch_size
        self.seq_len = args.seq_len
        self.checkpoint = args.checkpoint
        self.model_directory = args.model_directory

    def on_epoch_end(self, epoch, logs=None):
        """
        Apply the prediction of the model to a batch of data at the end of each epoch.
        """
        vocab = Vocabulary(self.vocabulary)
        X_batch, y_batch = next(generator(self.corpus, self.vocabulary,
                                          self.batch_size, self.seq_len))
        sample_prediction(self.model, X_batch, y_batch, vocab)
        path_to_model = os.path.join(self.model_directory, "keras_pred_e{}.h5".format(epoch))
        if epoch % self.checkpoint == 0:
            self.log.info("Saving the model to %s", path_to_model)
            self.model.save(path_to_model)


def train_predict(args, log_level=logging.INFO):
    """
    Train the model and show the progress of the prediction at each epoch.
    """
    _log = logging.getLogger("training")
    _log.setLevel(log_level)

    if args.from_model:
        model = load_model(args.from_model)
    else:
        model = generate_model(args)

    ON_EPOCH_END_CALLBACK = OnEpochEndCallback(args, _log)
    model.fit_generator(generator(args.corpus, args.vocabulary, args.batch_size, args.seq_len),
                        samples_per_epoch=args.steps_per_epoch,
                        nb_epoch=args.nb_epochs,
                        callbacks=[ON_EPOCH_END_CALLBACK, ],
                        validation_data=None)
