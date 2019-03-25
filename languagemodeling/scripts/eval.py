"""Evaulate a language model using a test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import math

from nltk.corpus import gutenberg
from nltk.corpus import PlaintextCorpusReader

if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    # WORK HERE!! LOAD YOUR EVALUATION CORPUS
    corpus = PlaintextCorpusReader('dataset', 'new-comments-test-10.txt')
    sents = corpus.sents()
    #sents = gutenberg.sents('austen-persuasion.txt')

    # compute the 
    # WORK HERE!!
    log_prob = None
    e = None
    p = None

    print('Log probability: {}'.format(log_prob))
    print('Cross entropy: {}'.format(e))
    print('Perplexity: {}'.format(p))
