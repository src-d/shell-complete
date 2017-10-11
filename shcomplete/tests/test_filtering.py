import argparse
import os
import tempfile
import unittest

from shcomplete.filtering import ShellFiltering, BashTimestamp, ZshTimestamp
from shcomplete.filtering import filter


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.min_nb_lines = 10
        self.bash_history = "shcomplete/tests/data/histories/bash_history"
        self.zsh_history = "shcomplete/tests/data/histories/.zsh_history"
        self.shell_filtering = ShellFiltering()
        self.bash_timestamp = BashTimestamp()
        self.zsh_timestamp = ZshTimestamp

    def test_too_small(self):
        self.assertFalse(self.shell_filtering.is_too_small(self.bash_history, self.min_nb_lines))

    def test_pgp_message(self):
        self.assertFalse(self.shell_filtering.is_pgp_message(self.bash_history))

    def test_HTML(self):
        self.assertFalse(self.shell_filtering.is_HTML(self.bash_history))

    def test_match(self):
        self.assertTrue(self.bash_timestamp.matches(self.bash_history))
        self.assertFalse(self.bash_timestamp.matches(self.zsh_history))

    def test_timestamps(self):
        self.assertFalse(self.bash_timestamp.detect_timestamps(self.bash_history))

    def test_comments(self):
        self.assertFalse(self.shell_filtering.detect_comments(self.bash_history))

    def test_remove_timestamp(self):
        zsh_timestamp_ex = ": 1416478680:0;"
        with tempfile.NamedTemporaryFile(prefix="shcomplete-test-zsh_timestamp",
                                         suffix=".txt", mode='w+') as tmpf:
            tmpf.write(zsh_timestamp_ex + "git add -u")
            tmpf.seek(0)
            self.assertTrue(self.zsh_timestamp.detect_timestamps(tmpf.name))
            self.zsh_timestamp.remove_timestamps(tmpf.name)
            self.assertFalse(self.zsh_timestamp.detect_timestamps(tmpf.name))

    def test_remove_comments(self):
        comment = "# whatever comment\n"
        with tempfile.NamedTemporaryFile(prefix="shcomplete-test-history_w_commments",
                                         suffix=".txt", mode='w+') as tmpf:
            tmpf.write(comment + "git add -u")
            tmpf.seek(0)
            self.assertTrue(self.shell_filtering.detect_comments(tmpf.name))
            self.shell_filtering.remove_superfluous(tmpf.name)
            self.assertFalse(self.shell_filtering.detect_comments(tmpf.name))

    def test_filter(self):
        args = argparse.Namespace(data_directory="shcomplete/tests/data/histories",
                                  min_nb_lines=10)
        filter(args)
        for root, d, files in os.walk(args.data_directory):
            for f in files:
                path_to_file = os.path.join(root, f)
                self.assertFalse(self.shell_filtering.is_too_small(path_to_file,
                                                                   args.min_nb_lines))
                self.assertFalse(self.shell_filtering.is_pgp_message(path_to_file))
                self.assertFalse(self.shell_filtering.is_HTML(path_to_file))
                self.assertFalse(self.shell_filtering.detect_comments(path_to_file))
                self.assertFalse(self.bash_timestamp.detect_timestamps(path_to_file))


if __name__ == '__main__':
    unittest.main()
