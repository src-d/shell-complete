import re
import os

__shells__ = []

def register_shell(cls):
    __shells__.append(cls)
    return cls

class Shell:
    @classmethod
    def matches(self, file_name):
        """
        Check which kind of shell it is : bash, zsh, or fish
        """
        return file_name.endswith(self.HISTORY_FILE)

    @classmethod
    def detect_timestamps(self, file_name):
        """
        Return a bool if there are timestamps to remove in file_name
        """
        with open(file_name, "r") as f:
            return self.TIMESTAMP_RE.match(f.readline())

    @classmethod
    def remove_timestamps(self, file_name):
        """
        Remove timestamps in file_name
        """
        with open(file_name, "r+") as f:
            f.seek(0)
            f.truncate()
            for line in f:
                if pattern in line:
                    command = cls.TIMESTAMP_RE.split(line)
                    if len(command) > 1:
                        f.write(command[1])

@register_shell
class Zsh(Shell):
    HISTORY_FILE = "zsh_history"
    TIMESTAMP_RE = re.compile(r": \d{10}:\d{1,};")

@register_shell
class Fish(Shell):
    HISTORY_FILE = "fish_history"
    TIMESTAMP_RE = re.compile(r"- cmd: ")

@register_shell
class Bash(Shell):
    HISTORY_FILE = "bash_history"
    TIMESTAMP_RE = re.compile(r" \d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} ")

def filter_timestamps(args):
    """
    Remove timestamps in all history files that need it
    """
    for root, d, files in os.walk("shcomplete/data"):
        for f in files:
            path_to_file = os.path.join(root, f)
            for cls in __shells__:
                if cls.matches(path_to_file) and cls.detect_timestamps(path_to_file):
                    cls.remove_timestamps(path_to_file)
