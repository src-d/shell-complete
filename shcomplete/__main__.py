import argparse
import logging
import sys

from modelforge.logs import setup_logging

from shcomplete.repos import fetch_repos
from shcomplete.filtering import filter
from shcomplete.corpus import write_corpus
from shcomplete.tfdf import filter_prediction_set


def one_arg_parser(*args, **kwargs) -> argparse.ArgumentParser:
    """
    Create parser for one argument with passed arguments.
    It is helper function to avoid argument duplication in subcommands.
    :return: Parser for one argument.
    """
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument(*args, **kwargs)
    return arg_parse


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

    data_directory_arg_default = one_arg_parser("-d", "--data-directoty", required=True,
                                                help="Path to input data.")
    output_arg_default = one_arg_parser("-o", "--output", required=True,
                                        help="Path to output file")

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
    filtering_parser.add_argument("--min-nb-lines", type=int, default=10,
                                  help="Minimum number of lines in each line.")

    tfdf_parser = subparsers.add_parser(
        "tfdf", help="Filter prefixes in all files based on their tfdf score."
        "Return the ones the model will be able to predict in the next command",
        parents=[data_directory_arg_default, output_arg_default])
    tfdf_parser.set_defaults(handler=filter_prediction_set)
    # Nodes in the trie increase exponentially with line lengths.
    tfdf_parser.add_argument("--max-length", type=int, default=8,
                             help="Maximum number of tokens in each line.")
    tfdf_parser.add_argument("--threshold", type=float, default=0.01,
                             help="Measure of noise in the dictionary of prefixes.")

    corpus_parser = subparsers.add_parser(
        "corpus", help="Write all the history files into an output txt file."
        "One file per line.",
        parents=[input_directory_arg_default, output_arg_default])
    corpus_parser.set_defaults(handler=write_corpus)

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
