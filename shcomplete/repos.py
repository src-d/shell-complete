import time
import os
import argparse
import sys
import logging
import re

from time import sleep
from github import Github as GitHub, RateLimitExceededException, GithubException

__shells__ = []

def register_shell(cls):
    __shells__.append(cls)
    return cls

class Shell:
    """
    Collects the list of repositories with either bash/zsh/fish history files
    """
    @classmethod
    def get_repos(self, api):
        """
        Get the list of repositories
        """
        logging.basicConfig(filename='repos.log', level=logging.INFO)

        repos = []
        query = api.search_code(self.GITHUB_QUERY)
        success = False
        while not success:
            try:
                query = api.search_code(self.GITHUB_QUERY)
                for rep in query:
                    repos.append(rep.html_url)
                success = True
            except GithubException:
                print("Hit rate limit, sleeping 60 seconds...")
                sleep(60)
                continue
            except Exception as e:
                print("type(%s): %s" % (type(e), e))
                sleep(0.1)
                continue

        repos = [x for x in repos if x.endswith(self.HISTORY_FILE)]

        return repos

def to_raw_urls(urls):
    """
    Transform the links returned by GitHub API, github.com into raw.githubusercontent.com
    """
    base = "https://raw.githubusercontent.com"
    raw_urls = []
    for url in urls:
        url = url.split("/")
        raw_url = base
        for i in range(3, len(url)):
            if url[i] != "blob":
                raw_url += "/" + url[i]
        raw_urls.append(raw_url)
    return raw_urls


@register_shell
class Zsh(Shell):
    HISTORY_FILE = re.compile("/.?zsh_history")
    GITHUB_QUERY = "filename:zsh_history -extension:zsh size:>0"

@register_shell
class Fish(Shell):
    HISTORY_FILE = "fish_history"
    GITHUB_QUERY = "filename:fish_history size:>0"

@register_shell
class Bash(Shell):
    HISTORY_FILE = re.compile("/.?bash_history")
    GITHUB_QUERY = "filename:bash_history size:>=950 -user:Dahs81"


def fetch_repos(args):
    """
    Fetch the repositories and write the raw links into an output txt file
    """
    g = GitHub(args.token, password=None, timeout=40, per_page=30)
    repos = []
    for cls in __shells__:
        repos += cls.get_repos(g)
    repos = to_raw_urls(repos)
    with open(args.output, 'w') as f:
        for rep in repos:
            f.write(rep + '\n')
