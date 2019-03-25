# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math


def addmarks(sent, n):
    """
    Add start and end markers.

    input sent, n.
    """
    sent = ["<s>"] * (n - 1) + sent + ["</s>"]
    return sent


class LanguageModel(object):
    """Language Model."""

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return 0.0

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return -math.inf

    def log_prob(self, sents):
        """Log-probability of a list of sentences.

        sents -- the sentences.
        """
        # WORK HERE!!

    def cross_entropy(self, sents):
        """Cross-entropy of a list of sentences.

        sents -- the sentences.
        """
        # WORK HERE!!

    def perplexity(self, sents):
        """Perplexity of a list of sentences.

        sents -- the sentences.
        """
        # WORK HERE!!


class NGram(LanguageModel):
    """Ngram model."""

    def __init__(self, n, sents):
        """
        Input, n -- order of the model.

        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self._n = n

        count = defaultdict(int)

        for sent in sents:
            sent = addmarks(sent, n)
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                count[ngram] += 1
                count[ngram[:-1]] += 1

        self._count = dict(count)

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        prob = 0

        # Si prev_tokens es None devuelvo una tupla vacia
        # si no una tupla con los prev_tokens
        prev_tokens = tuple(prev_tokens) if prev_tokens else tuple()

        tokens = prev_tokens + (token,)

        count_prev_tokens = self.count(tuple(prev_tokens))

        if count_prev_tokens != 0:
            prob = float(self.count(tuple(tokens)) / count_prev_tokens)
        return prob

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        n = self._n

        sent = addmarks(sent, n)

        prob = 1
        for i in range(n - 1, len(sent)):
            prob *= self.cond_prob(sent[i], sent[i - n + 1: i])
        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        n = self._n
        # add markers inicio y fin
        sent = addmarks(sent, n)
        prob = 0
        for i in range(n - 1, len(sent)):
            # sent[i] token
            # sent[i - n + 1: i] prev_tokens
            cond_prob = self.cond_prob(sent[i], sent[i - n + 1: i])
            # aplico log sobre las probabilidades y las sumo
            prob += math.log(cond_prob)
        return prob


class AddOneNGram(NGram):
    """Add-one estimation (Laplace smoothing)."""

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        # call superclass to compute counts
        super().__init__(n, sents)

        # compute vocabulary
        self._voc = voc = set()
        # WORK HERE!!

        self._V = len(voc)  # vocabulary size

    def V(self):
        """Size of the vocabulary.
        """
        return self._V

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # WORK HERE!!


class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self._n = n

        if gamma is not None:
            # everything is training data
            train_sents = sents
        else:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        print('Computing counts...')
        # WORK HERE!!
        # COMPUTE COUNTS FOR ALL K-GRAMS WITH K <= N

        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            print('Computing vocabulary...')
            self._voc = voc = set()
            # WORK HERE!!

            self._V = len(voc)

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')
            # WORK HERE!!
            # use grid search to choose gamma

    def count(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        # WORK HERE!! (JUST A RETURN STATEMENT)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # WORK HERE!!
