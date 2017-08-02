import argparse
import logging
import sys

from modelforge.logs import setup_logging

from shcomplete.repos import fetch_repos
from shcomplete.filtering import filter
from shcomplete.corpus import write_corpus
from shcomplete.tfdf import filter_prediction_set


def main():
    """
    Creates all the argparse-rs and invokes the function from set_defaults().
    :return: The result of the function from set_defaults().
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO",
                        choices=logging._nameToLevel,
                        help="Logging verbosity.")
    subparsers = parser.add_subparsers(help="Commands.", dest="command")

    repos_parser = subparsers.add_parser(
        "repos", help="Return the list of repositories with history files.")
    repos_parser.set_defaults(handler=fetch_repos)
    repos_parser.add_argument(
        "-t", "--token", required=True,
        help="Github API token.")
    repos_parser.add_argument("--timeout", type=int, default=100,
                              help="Maximum time the Search API can run.")
    repos_parser.add_argument("--per-page", type=int, default=100,
                              help="Number of repositories returned per page by the Search API.")
    # Since the API can return slightly different results, running it many times give more results
    repos_parser.add_argument("--nb-search", type=int, default=5,
                              help="Number of times we launch the Search API.")
    repos_parser.add_argument(
        "-o", "--output", required=True,
        help="Path to the output file.")

    filtering_parser = subparsers.add_parser(
        "filtering", help="Remove timestamps in history files.")
    filtering_parser.add_argument(
        "-d", "--directory", required=True,
        help="Path to the data.")
    filtering_parser.add_argument("--min-nb-lines", type=int, default=10,
                                  help="Minimum number of lines in each line.")
    filtering_parser.set_defaults(handler=filter)

    tfdf_parser = subparsers.add_parser(
        "tfdf", help="Filter prefixes in all files based on their tfdf score."
        "Return the ones the model will be able to predict in the next command")
    tfdf_parser.add_argument(
        "-d", "--directory", required=True,
        help="Path to the data.")
    # Nodes in the trie increase exponentially with line lengths.
    tfdf_parser.add_argument("--max-length", type=int, default=8,
                             help="Maximum number of tokens in each line.")
    tfdf_parser.add_argument("--threshold", type=float, default=0.01,
                             help="Measure of noise in the dictionary of prefixes.")
    tfdf_parser.add_argument(
        "-o", "--output", required=True,
        help="Path to the output file containing the dictionary.")
    tfdf_parser.set_defaults(handler=filter_prediction_set)

    corpus_parser = subparsers.add_parser(
        "corpus", help="Write all the history files into an output txt file."
        "One file per line.")
    corpus_parser.set_defaults(handler=write_corpus)
    corpus_parser.add_argument(
        "-d", "--directory", default="shcomplete/data",
        help="Path to the data.")
    corpus_parser.add_argument(
        "-o", "--output", required=True,
        help="Path to the output file.")

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
