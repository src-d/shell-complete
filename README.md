# Shell Complete - WIP [![Build Status](https://travis-ci.org/src-d/shell-complete.svg)](https://travis-ci.org/src-d/shell-complete) [![codecov](https://codecov.io/github/src-d/shell-complete/coverage.svg?branch=master)](https://codecov.io/gh/src-d/shell-complete) [![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

Sequential [Keras](https://keras.io/) models for both command line misprints correction and next command prediction. The RNNs are trained on datasets of bash/zsh/fish history files gathered from GitHub.

## Installation

```
pip3 install git+https://github.com/src-d/shell-complete
```

## Run the data pipeline

* Get the list of repositories

To use the GitHub API, you need to generate a personel access token, see [GitHub help](https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/). Then, run:

```
shcomplete repos -t token -o output.txt
```

* Get the history files using [Scrapy](https://scrapy.org/)

```
scrapy runspider repospider.py
```

* Clean the dataset

```
shcomplete filtering -d shcomplete/data
```

* Build a vocabulary of command line prefixes based on TF-DF score

Store command line prefixes into a trie data structure, using [google/pygtrie](https://github.com/google/pygtrie
). Compute the Term-Frequency Document-Frequency score of each prefix and prune the trie based on these numerical statistics to keep only the relevant prefixes. The level of noise in this vocabulary depends on the threshold parameter.

```
shcomplete tfdf -d shcomplete/data -o vocabulary.txt
```

* Build the corpus, input when generating batches of data

```
shcomplete corpus -d shcomplete/srcd -o output.txt
```

## Train the sequential Keras models

See the following command line interface to train the RNNs for both misprints correctionon and next command prediction, on the previous dataset of command line histories.

```
shcomplete model2correct --help
shcomplete model2predict --help
```

As regards misprints correction, a sequential model that reached 99% accuracy on more than 1000 basic command line prefixes after 100 epochs with 4 GPUs is provided in /saved_models. If you want it to take into account your aliases or specific commands, we recommand you to train this model on your own history.

## Usage - WIP

## License

Apache 2.0.
