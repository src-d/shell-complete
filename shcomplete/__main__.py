import argparse
import logging
import sys

from modelforge.logs import setup_logging
from keras.layers import recurrent

from shcomplete.repos import fetch_repos
from shcomplete.filtering import filter
from shcomplete.corpus import write_corpus
from shcomplete.tfdf import filter_prediction_set
from shcomplete.model2predict import train_predict
from shcomplete.model2correct import train_correct


def one_arg_parser(*args, **kwargs) -> argparse.ArgumentParser:
    """
    Create parser for one argument with passed arguments.
    It is helper function to avoid argument duplication in subcommands.
    :return: Parser for one argument.
    """
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument(*args, **kwargs)
    return arg_parser


def get_parser() -> argparse.ArgumentParser:
    """
    Create main parser.
    :return: Parser
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO",
                        choices=logging._nameToLevel,
                        help="Logging verbosity.")

    # Create all common arguments

    data_directory_arg_default = one_arg_parser("-d", "--data-directory", required=True,
                                                help="Path to input data.")
    output_arg_default = one_arg_parser("-o", "--output", required=True,
                                        help="Path to output file")
    corpus_arg_default = one_arg_parser("--corpus", required=True,
                                        help="Path to the corpus of shell histories."
                                        "Text file where histories are delimited by a blank line.")
    vocabulary_arg_default = one_arg_parser("--vocabulary", required=True,
                                            help="Path to the file that contains the vocabulary.")
    cell_type_arg_default = one_arg_parser("--cell-type", default=recurrent.LSTM,
                                           help="Base type of cells for recurrent layers")
    input_layers_arg_default = one_arg_parser("--input-layers", type=int, default=2,
                                              help="Nuber of input layers.")
    hidden_layers_arg_default = one_arg_parser("--hidden-layers", type=int, default=128,
                                               help="Number of neurons in the network.")
    output_layers_arg_default = one_arg_parser("--output-layers", type=int, default=1,
                                               help="Number of output layers.")
    batch_size_arg_default = one_arg_parser("--batch-size", type=int, default=32,
                                            help="Number of samples we use for model training.")
    nb_epochs_arg_default = one_arg_parser("--nb-epochs", type=int, default=500,
                                           help="Number of iterations on the data.")
    steps_per_epoch_arg_default = one_arg_parser("--steps-per-epoch", type=int, default=16384,
                                                 help="Number of steps to yield from generator"
                                                 "before moving from one epoch to another.")
    dropout_arg_default = one_arg_parser("--dropout", type=float, default=0.4,
                                         help="Randomly turn off some fraction of neurons"
                                         "on each training iteration.")
    optimizer_arg_default = one_arg_parser("--optimizer", type=str, default="adam",
                                           help="Optimizer required for compiling the Keras model")
    models_directory_arg_default = one_arg_parser("--models-directory", required=True,
                                                  help="Directory where models are saved"
                                                  "at any checkpoint.")
    from_model_arg_default = one_arg_parser("--from-model",
                                            help="Path to the model to start the training from.")
    checkpoint_arg_default = one_arg_parser("--checkpoint", type=int, default=100,
                                            help="Save the model at each epoch multiple of"
                                            "the checkpoint number.")

    # Create and construct subparsers

    subparsers = parser.add_subparsers(help="Commands.", dest="command")

    repos_parser = subparsers.add_parser(
        "repos", help="Return the list of repositories with history files.",
        parents=[output_arg_default])
    repos_parser.set_defaults(handler=fetch_repos)
    repos_parser.add_argument(
        "-t", "--token", required=True,
        help="Github API token.")
    repos_parser.add_argument("--timeout", type=int, default=100,
                              help="GitHub Search API timeout - longer requests are dropped.")
    repos_parser.add_argument("--per-page", type=int, default=100,
                              help="Number of repositories returned per page by the Search API.")
    repos_parser.add_argument("--nb-search", type=int, default=5,
                              help="Number of times we launch the Search API.")

    filtering_parser = subparsers.add_parser(
        "filtering", help="Remove timestamps in history files.",
        parents=[data_directory_arg_default])
    filtering_parser.set_defaults(handler=filter)
    filtering_parser.add_argument("--min-nb-lines", type=int, default=50,
                                  help="Minimum number of lines in each line.")

    tfdf_parser = subparsers.add_parser(
        "tfdf", help="Filter prefixes in all files based on their tfdf score."
        "Return the ones the model will be able to predict in the next command.",
        parents=[data_directory_arg_default, output_arg_default])
    tfdf_parser.set_defaults(handler=filter_prediction_set)
    tfdf_parser.add_argument("--max-length", type=int, default=8,
                             help="Maximum number of tokens in each line.")
    tfdf_parser.add_argument("--threshold", type=float, default=0.01,
                             help="Measure of noise in the dictionary of prefixes.")

    corpus_parser = subparsers.add_parser(
        "corpus", help="Write all the history files into an output txt file."
        "One file per line.",
        parents=[data_directory_arg_default, output_arg_default])
    corpus_parser.set_defaults(handler=write_corpus)

    model2predict_parser = subparsers.add_parser(
        "model2predict", help="Train a Keras sequential model."
        "Return at the end of each epoch, a random sequence of commands and its prediction.",
        parents=[vocabulary_arg_default, corpus_arg_default, checkpoint_arg_default,
                 models_directory_arg_default, from_model_arg_default, batch_size_arg_default,
                 input_layers_arg_default, hidden_layers_arg_default, output_layers_arg_default,
                 nb_epochs_arg_default, steps_per_epoch_arg_default, dropout_arg_default,
                 optimizer_arg_default, cell_type_arg_default])
    model2predict_parser.set_defaults(handler=train_predict)
    model2predict_parser.add_argument("--seq-len", type=int, default=50,
                                      help="Length of the input sequence"
                                      "we want to predict the next command.")

    model2correct_parser = subparsers.add_parser(
        "model2correct", help="Train a Keras sequential model."
        "Return at the end of each epoch, a random misspelled command and its correction.",
        parents=[vocabulary_arg_default, corpus_arg_default, checkpoint_arg_default,
                 models_directory_arg_default, from_model_arg_default, batch_size_arg_default,
                 input_layers_arg_default, hidden_layers_arg_default, output_layers_arg_default,
                 nb_epochs_arg_default, steps_per_epoch_arg_default, dropout_arg_default,
                 optimizer_arg_default, cell_type_arg_default])
    model2correct_parser.set_defaults(handler=train_correct)
    model2correct_parser.add_argument("--max-cmd-len", type=int, default=40,
                                      help="Maximum number of characters"
                                      "in the command input.")
    model2correct_parser.add_argument("--level-noise", type=float, default=0.4,
                                      help="level of noise when generating"
                                      "misspelling mistakes in command lines.")
    model2correct_parser.add_argument("--nb-predictions", type=int, default=10,
                                      help="Number of predictions printed"
                                      "at the end of each epoch.")

    return parser


def main():
    """
    Creates all the argparse-rs and invokes the function from set_defaults().
    :return: The result of the function from set_defaults().
    """

    parser = get_parser()
    args = parser.parse_args()
    args.log_level = logging._nameToLevel[args.log_level]
    setup_logging(args.log_level)

    try:
        handler = args.handler
    except AttributeError:
        def print_usage(_):
            parser.print_usage()

        handler = print_usage
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
