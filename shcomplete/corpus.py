import os


def write_file(args, path_to_file, file_delimiter="FILE DELIMITER\n"):
    """
    Write the history located to path_to_file.
    """
    with open(args.output, "a") as fout:
        with open(path_to_file, "r") as f:
            for line in f:
                fout.write(line)
        fout.write(file_delimiter)


def write_corpus(args):
    """
    Write all the history files into an text file, called corpus.
    The histories are separated in the corpus by a file_delimiter.
    """
    nb_file = 0
    for root, d, files in os.walk(args.data_directory):
        for f in files:
            nb_file += 1
            path_to_file = os.path.join(root, f)
            write_file(args, path_to_file)
