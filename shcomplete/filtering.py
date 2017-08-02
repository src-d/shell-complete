import os
import re

__shells__ = []


def register_shell(cls):
    __shells__.append(cls)
    return cls


class ShellFiltering:

    COMMENT_RE = re.compile(r"^#")
    PGP_MESSAGE = re.compile(r"-----BEGIN PGP MESSAGE-----")
    HTML_DOCTYPE = re.compile(r"<\!DOCTYPE html>")

    @classmethod
    def is_too_small(cls, file_name, min_nb_lines):
        """
        Check if file_name contains less than 10 lines.
        """
        with open(file_name, "r") as f:
            lines = f.readlines()
            return len(lines) < min_nb_lines

    @classmethod
    def is_pgp_message(cls, file_name):
        """
        Check if file_name is actually a PGP message and not a history file.
        Example of PGP message to remove :
        https://github.com/hackerunion/root/blob/master/home/ricka/.bash_history
        """
        with open(file_name, "r") as f:
            return cls.PGP_MESSAGE.match(f.readline())

    @classmethod
    def is_HTML(cls, file_name):
        """
        Check if type of file_name is HTML.
        """
        with open(file_name, "r") as f:
            return cls.HTML_DOCTYPE.match(f.readline())

    @classmethod
    def matches(cls, file_name):
        """
        Check which kind of shell it is : bash, zsh, or fish
        """
        return file_name.endswith(cls.HISTORY_FILE)

    @classmethod
    def detect_timestamps(cls, file_name):
        """
        Return a bool if there are timestamps to remove in file_name
        """
        with open(file_name, "r") as f:
            return cls.TIMESTAMP_RE.match(f.readline())

    @classmethod
    def detect_comments(cls, file_name):
        """
        Return a bool if there are comments to remove in file_name
        """
        with open(file_name, "r") as f:
            return cls.COMMENT_RE.match(f.readline())

    @classmethod
    def remove_timestamps(cls, file_name):
        """
        Remove timestamps in file_name
        """
        with open(file_name, "r+") as f:
            lines = f.readlines()
            f.seek(0)
            f.truncate()
            for line in lines:
                if cls.TIMESTAMP_RE.search(line):
                    command = cls.TIMESTAMP_RE.split(line)
                    if len(command) > 1:
                        f.write(command[1])

    @classmethod
    def remove_superfluous(cls, file_name):
        """
        Remove comments, empty lines and lines starting with one element in file_name
        """
        with open(file_name, "r+") as f:
            lines = f.readlines()
            f.seek(0)
            f.truncate()
            for line in lines:
                line = line.rstrip()
                first_token = line.split(" ")[0]
                if len(first_token) > 1 and not cls.COMMENT_RE.search(line):
                    f.write(line + "\n")


@register_shell
class ZshTimestamp(ShellFiltering):
    HISTORY_FILE = "zsh_history"
    TIMESTAMP_RE = re.compile(r": \d{10}:\d{1,};")


@register_shell
class FishTimestamp(ShellFiltering):
    HISTORY_FILE = "fish_history"
    TIMESTAMP_RE = re.compile(r"- cmd: ")


@register_shell
class BashTimestamp(ShellFiltering):
    HISTORY_FILE = "bash_history"
    TIMESTAMP_RE = re.compile(r" \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} ")


def filter(args):
    """
    Remove timestamps in all history files that need it
    """
    min_nb_lines = args.min_nb_lines

    for root, d, files in os.walk(args.data_directory):
        for f in files:
            path_to_file = os.path.join(root, f)
            if ShellFiltering.is_pgp_message(path_to_file) or ShellFiltering.is_HTML(path_to_file):
                os.remove(path_to_file)
            else:
                for cls in __shells__:
                    if cls.matches(path_to_file) and cls.detect_timestamps(path_to_file):
                        cls.remove_timestamps(path_to_file)
                ShellFiltering.remove_superfluous(path_to_file)
                if ShellFiltering.is_too_small(path_to_file, min_nb_lines):
                    os.remove(path_to_file)
