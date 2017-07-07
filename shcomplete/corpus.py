import os
import logging

def write_file(args, path_to_file):
    """
    Write path_to_file on 1 line of output file
    """
    with open(args.output, "a") as fout:
        with open(path_to_file, "r") as f:
            for line in f:
                line = line.rstrip() + " "
                fout.write(line)
            fout.write("\ñ")
            fout.write("\n")


def write_corpus(args):
    """
    Write all the history files into an output txt file. One file per line.
    """
    logging.basicConfig(filename='corpus.log', level=logging.INFO)

    nb_file=0
    for root, d, files in os.walk("shcomplete/data"):
        for f in files:
            nb_file += 1
            path_to_file = os.path.join(root, f)
            logging.info("file nº{x} : \n{y}".format(x=nb_file, y=path_to_file))
            write_file(args, path_to_file)
