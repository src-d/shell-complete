import argparse
import logging
import sys

from modelforge.logs import setup_logging

from shcomplete.repos import fetch_repos
from shcomplete.filtering import filter_timestamps
from shcomplete.corpus import write_corpus


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
    repos_parser.add_argument(
        "-o", "--output", required=True,
        help="Path to the output file.")

    filtering_parser = subparsers.add_parser(
        "filtering", help="Remove timestamps in history files.")
    filtering_parser.add_argument(
        "-d", "--directory", default="shcomplete/data",
        help="Path to the data.")
    filtering_parser.set_defaults(handler=filter_timestamps)

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
