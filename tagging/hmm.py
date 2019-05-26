import math
import itertools
from collections import defaultdict


def log2ext(x):
    """Extend log function.

    if x = 0 ==> -inf
    if x != 0 ==> log2(x)
    """
    return (lambda x: math.log(x, 2) if x > 0 else float('-inf'))(x)


def addmarks(sent, n):
    """
    Add start and end markers.

    input sent, n.
    """
    sent = ["<s>"] * (n - 1) + sent + ["</s>"]
    return sent


class HMM:
    """Hidden Markov Models."""

    def __init__(self, n, tagset, trans, out):
        """
        n -- n-gram size.
        tagset -- set of tags.
        trans -- transition probabilities dictionary.
        out -- output probabilities dictionary.
        """
        self._n = n
        self._tagset = tagset
        self._trans = trans
        self._out = out

    def tagset(self):
        """Returns the set of tags.
        """
        return self._tagset

    def trans_prob(self, tag, prev_tags):
        """Probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        return self._trans.get(prev_tags, {}).get(tag, 0.0)

    def out_prob(self, word, tag):
        """Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        return self._out.get(tag, {}).get(word, 0.0)

    def tag_prob(self, y):
        """
        Probability of a tagging.
        Warning: subject to underflow problems.

        y -- tagging.
        """

        n = self._n
        y = addmarks(y, n)
        tag_length = len(y)

        tag_prob = 1
        for i in range(n - 1, tag_length):
            tag = y[i]
            prev_tags = y[i - n + 1: i]
            tag_prob *= self.trans_prob(tag, tuple(prev_tags))

        return tag_prob

    def prob(self, x, y):
        """
        Joint probability of a sentence and its tagging.
        Warning: subject to underflow problems.

        x -- sentence.
        y -- tagging.
        """
        # check length of sentece and tagging
        assert len(x) == len(y)

        q_prob = self.tag_prob(y)
        e_prob = 1 
        for word, tag in zip(x, y):
            e_prob *= self.out_prob(word, tag)

        return q_prob * e_prob

    def tag_log_prob(self, y):
        """
        Log-probability of a tagging.

        y -- tagging.
        """
        n = self._n
        y = addmarks(y, n)
        tag_length = len(y)

        tag_log_prob = 0
        for i in range(n - 1, tag_length):
            tag = y[i]
            prev_tags = y[i - n + 1: i]
            tag_log_prob += log2ext(self.trans_prob(tag, tuple(prev_tags)))
        return tag_log_prob

    def log_prob(self, x, y):
        """
        Joint log-probability of a sentence and its tagging.

        x -- sentence.
        y -- tagging.
        """
        assert len(x) == len(y)
        log_q_prob = self.tag_log_prob(y)
        log_e_prob = 0
        for word, tag in zip(x, y):
            log_e_prob += log2ext(self.out_prob(word, tag))

        return log_q_prob + log_e_prob

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        return ViterbiTagger(self).tag(sent)


class ViterbiTagger:

    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """
        self._hmm = hmm

    def tag(self, sent):
        """Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        hmm = self._hmm
        n = hmm._n
        tagset = hmm._tagset
        # table for viterbi algorithm
        # pi = { key : { prev_tags : (log_prob, list_tags) } }
        self._pi = pi = defaultdict(lambda: defaultdict(tuple))
        # initilization
        # from book pi(0, *, * ) = 1
        pi[0][("<s>",) * (n - 1)] = (log2ext(1.0), [])

        length_sent = len(sent)

        # build table
        for i in range(1, length_sent + 1):
            word = sent[i - 1]
            for tag in tagset:
                # e(word|tag) from book e(x_k | v)
                # P(word | tag)
                prob_word_tag = hmm.out_prob(word, tag)
                if prob_word_tag > 0.:
                    # check previous column
                    for prev_tags, (log_prob, list_tags) in pi[i - 1].items():
                        # q(tag | prev_tags) from book q(v | w, u)
                        q_prob = hmm.trans_prob(tag, prev_tags)

                        if q_prob > 0.:
                            # from book pi(x-1, w, u) x q()
                            # new_log_prob = log_prob * q_prob * prob_word_tag
                            new_log_prob = (log_prob + log2ext(q_prob) +
                                            log2ext(prob_word_tag))
                            
                            new_list_tags = list_tags + [tag]
                            
                            new_prev_tags = (prev_tags + tuple(tag,))[1:]
                            # print("LIST tags",list_tags, type(list_tags))
                            # print('-----')
                            # print("Tag", tag, type(tag))
                            # print('-----')
                            # print("NEW LIST OF TAGS ", new_list_tags, type(new_list_tags))
                            # look for the tag which maximize the pro
                            # or if new_prev_tags not in the table, add it.
                            if ((new_prev_tags not in pi[i-1]) or (new_log_prob > pi[i-1][new_prev_tags][0])):
                                pi[i][new_prev_tags] = (new_log_prob, new_list_tags)
        # return max prob
        max_prob = float('-inf')
        best_tags = []
        print(dict(pi))
        # check all the posible list of tags for length sent
        for prev_tags, (log_prob, list_tags) in pi[length_sent].items():
            # from book q(STOP | u, v)
            q_prob_stop = hmm.trans_prob("</s>", prev_tags)
            # new_log_prob = log_prob * q_prob_stop
            new_log_prob = log_prob + log2ext(q_prob_stop)
            if new_log_prob > max_prob:
                max_prob = new_log_prob
                best_tags = list_tags

        return best_tags
