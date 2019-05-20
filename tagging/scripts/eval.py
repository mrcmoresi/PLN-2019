"""Evaulate a tagger.

Usage:
  eval.py -i <file> [-c]
  eval.py -h | --help

Options:
  -c            Show confusion matrix.
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import sys
from collections import defaultdict, Counter

from sklearn.metrics import confusion_matrix
from tagging.ancora import SimpleAncoraCorpusReader

import matplotlib.pyplot as plt


def progress(msg, width=None):
    """Ouput the progress of something on the same line."""
    if not width:
        width = len(msg)
    print('\b' * width + msg, end='')
    sys.stdout.flush()


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora-dataset/ancora-3.0.1es/', files)
    sents = list(corpus.tagged_sents())

    # tag and evaluate
    # WORK HERE!!
    hits = 0
    total = 0

    knw_hits, knw_total = 0, 0
    unk_hits, unk_total = 0, 0

    err_count = defaultdict(lambda: defaultdict(int))
    err_sent = defaultdict(lambda: defaultdict(set))

    n = len(sents)

    for i, sent in enumerate(sents):
        word_sent, gold_tag_sent = zip(*sent)
        tagged_sent = model.tag(word_sent)
        assert len(tagged_sent) == len(gold_tag_sent), i

        # check hits per sent
        sent_hits = [gold_tag == model_tag for gold_tag, model_tag in zip(gold_tag_sent, tagged_sent)]
        hits += sum(sent_hits)
        total += len(sent)
        accuracy = float(hits) / total

        # acc over unknown words
        unk_hits_sent = [hs for word, hs in zip(word_sent, sent_hits) if model.unknown(word)]
        unk_hits += sum(unk_hits_sent)
        unk_total += len(unk_hits_sent)
        unk_acc = unk_hits / unk_total

        # acc over known words
        if total == unk_total:
            known_acc = 0.0
        else:
            known_acc = (hits - unk_hits) / (total - unk_total)

        for tag1, tag2 in zip(tagged_sent, gold_tag_sent):
            err_count[tag2][tag1] += 1
            if tag2 != tag1:
                # keep index of miss tagged sent
                err_sent[tag2][tag1].add(i)

        format_str = ' i {} (Accuracy {:.2f}% / Known {:.2f}% / Unknown {:.2f}%)'
        progress(format_str.format(i, accuracy, known_acc, unk_acc))

    accuracy_global = hits / total
    # print(err_sent)
    if total == unk_total:
        known_acc_glob = 0.0
    else:
        known_acc_glob = (hits - unk_hits) / (total - unk_total)

    unk_acc_glob = unk_hits / unk_total
    print('\n Accuracy: {:.2f}% / Known {:.2f}% / Uknown {:.2f}%'.format(
        accuracy_global, known_acc_glob, unk_acc_glob))

    if opts['-c']:
        print('\nConfusion Matrix')
        print('================')
        # select most frequent tags
        most_freq_tag = sorted(err_count.keys(),
                               key=lambda tag: -sum(err_count[tag].values()))
        tags = most_freq_tag[:10]
        print('G \ M', end='')
        for tag in tags:
            print('\t{}'.format(tag), end='')
        print('')

        for tag1 in tags:
            print('{}\t'.format(tag1), end='')
            for tag2 in tags:
                if err_count[tag1][tag2] > 0:
                    acc = err_count[tag1][tag2] / total
                    print('{:.3f}\t'.format(acc), end='')
                else:
                    print('-\t'.format(acc), end='')
            print('')
