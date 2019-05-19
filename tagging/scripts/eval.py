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
from collections import defaultdict

from tagging.ancora import SimpleAncoraCorpusReader


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

    tags_gold_standard, tags_annotated = [], []
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

        format_str = ' i {} (Accuracy {:2.2f}% / Known {:2.2f}% / Unknown {:2.2f}%)'
        progress(format_str.format(i, accuracy, known_acc, unk_acc))

    accuracy_global = hits / total

    if total == unk_total:
        known_acc_glob = 0.0
    else:
        known_acc_glob = (hits - unk_hits) / (total - unk_total)

    unk_acc_glob = unk_hits / unk_total
    print('\n Accuracy: {:2.2f}% / Known {:2.2f}% / Uknown {:2.2f}%'.format(
        accuracy_global, known_acc_glob, unk_acc_glob))
