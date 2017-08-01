import time
import os
import argparse
import sys
import logging
import re

from time import sleep
from github import Github as GitHub
from github import GithubException

__shells__ = []


def register_shell(cls, log_level=logging.INFO):
    __shells__.append(cls)
    cls._log = logging.getLogger("repos")
    cls._log.setLevel(log_level)
    return cls


class Shell:
    """
    Collects the list of repositories with either bash/zsh/fish history files
    """

    @classmethod
    def get_repos(cls, api, level=logging.INFO):
        """
        Get the list of repositories
        """
        repos = []
        nb_repos = 0
        query = api.search_code(cls.GITHUB_QUERY)
        success_query = False
        while not success_query:
            try:
                for rep in query:
                    nb_repos += 1
                    repos.append(rep.html_url)
                    cls._log.info(
                        "repo nº{x} :\n{y}".format(x=nb_repos, y=rep.html_url))
                success_query = True
            except GithubException:
                print("Hit rate limit, sleeping 60 seconds...")
                sleep(60)
                continue
            except Exception as e:
                print("type(%s): %s" % (type(e), e))
                sleep(0.1)
                continue

        repos = [x for x in repos if cls.HISTORY_FILE.search(x)]
        repos = list(set(repos))  # remove duplicates

        return repos


def to_raw_urls(urls):
    """
    Transform the links returned by GitHub API :
    github.com into raw.githubusercontent.com.
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
    HISTORY_FILE = re.compile(r"/.?zsh_history$")
    GITHUB_QUERY = "filename:zsh_history -extension:zsh size:>0"


@register_shell
class Fish(Shell):
    HISTORY_FILE = re.compile(r"/.?fish_history$")
    GITHUB_QUERY = "filename:fish_history size:>0"


@register_shell
class Bash(Shell):
    HISTORY_FILE = re.compile(r"/.?bash_history$")
    GITHUB_QUERY = "filename:bash_history size:>0 -user:Dahs81"


def fetch_repos(args):
    """
    Fetch the repositories and write the raw links into an output txt file
    """
    g = GitHub(args.token, password=None, timeout=40, per_page=50)
    repos = []
    for cls in __shells__:
        repos += cls.get_repos(g)
    repos = to_raw_urls(repos)
    with open(args.output, "w") as f:
        for rep in repos:
            f.write(rep + "\n")
