"""Statistical Analysis dataset.

Usage:
  stats.py -d <dir>
  stats.py -h | --help

Options:
  -d <dir>   Dataset Directory.
  -h --help     Show this screen.

"""
import os

from collections import Counter
from docopt import docopt
from sentiment.tass import InterTASSReader


if __name__ == "__main__":
    opts = docopt(__doc__)

    directory = opts['-d']

    files = os.listdir(str(directory))
    print("Statistical Dataset Analysis")
    for dire in files:
        sub_dir = os.listdir(directory + dire)
        for f in sub_dir:
            reader = InterTASSReader(str(directory + '/' + dire + '/' + f))
            X, y = list(reader.X()), list(reader.y())
            print("Dataset {}".format(f))
            print("Total Amount of Tweets: {}".format(len(X)))
            for item in Counter(y).items():
                print(item[0], item[1])
            print('\n')
