import os


def write_file(args, path_to_file, line_delimiter=""):
    """
    Write path_to_file on 1 line of output file
    """
    with open(args.output, "a") as fout:
        with open(path_to_file, "r") as f:
            for line in f:
                line = line.rstrip() + " "
                fout.write(line)
                fout.write(line_delimiter)
            fout.write("\n")


def write_corpus(args):
    """
    Write all the history files into an output txt file. One file per line.
    """
    nb_file = 0
    for root, d, files in os.walk(args.directory):
        for f in files:
            nb_file += 1
            path_to_file = os.path.join(root, f)
            write_file(args, path_to_file)
