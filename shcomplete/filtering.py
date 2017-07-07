import re
import os

__shells__ = []


def register_shell(cls):
    __shells__.append(cls)
    return cls


class ShellTimestamp:

    COMMENT_RE = re.compile(r"^#")
    MOTIF_RE = re.compile(r"^ \d{1,}  ")

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
    def remove_timestamps(cls, file_name):
        """
        Remove timestamps in file_name
        """
        with open(file_name, "r+") as f:
            f.seek(0)
            f.truncate()
            for line in f:
                if ShellTimestamp.COMMENT_RE.search(line) or ShellTimestamp.MOTIF_RE.search(line):
                    f.write("")
                elif cls.TIMESTAMP_RE.search(line):
                    command = cls.TIMESTAMP_RE.split(line)
                    if len(command) > 1:
                        f.write(command[1])


@register_shell
class ZshTimestamp(ShellTimestamp):
    HISTORY_FILE = "zsh_history"
    TIMESTAMP_RE = re.compile(r": \d{10}:\d{1,};")


@register_shell
class FishTimestamp(ShellTimestamp):
    HISTORY_FILE = "fish_history"
    TIMESTAMP_RE = re.compile(r"- cmd: ")


@register_shell
class BashTimestamp(ShellTimestamp):
    HISTORY_FILE = "bash_history"
    TIMESTAMP_RE = re.compile(r" \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} ")


def filter_timestamps(args):
    """
    Remove timestamps in all history files that need it
    """
    for root, d, files in os.walk(args.directory):
        for f in files:
            path_to_file = os.path.join(root, f)
            for cls in __shells__:
                if (cls.matches(path_to_file) and cls.detect_timestamps(path_to_file)):
                    cls.remove_timestamps(path_to_file)
