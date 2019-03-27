from collections import defaultdict
import random


class NGramGenerator(object):
    """Ngram generator model."""

    def __init__(self, model):
        """Model -- n-gram model."""
        self._n = model._n

        # compute the probabilities
        probs = defaultdict(dict)
        counts = model._count
        for tokens in counts.keys():
            if len(tokens) == self._n:
                token = tokens[-1]
                prev_tokens = tokens[:-1]
                probs[prev_tokens][token] = model.cond_prob(token, prev_tokens)

        self._probs = dict(probs)

        # sort in descending order for efficient sampling
        # self._sorted_probs = sorted_probs = {}

        sorted_probs = {}
        for token, prob in probs.items():
            sorted_probs[token] = sorted(
                prob.items(), key=lambda item: item[1])

        self._sorted_probs = sorted_probs

    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self._n
        # arranco un sent vacio
        sent = []
        # indicador de inicio de sentencia
        prev_tokens = ['<s>'] * (n - 1)
        #genero un primer token
        token = self.generate_token(prev_tokens)
        while token != '</s>':
            sent.append(token)
            # junto los tokens pero solo veo a partir del segundo token
            prev_tokens = (prev_tokens + [token])[1:]
            # genero un token nuevo
            token = self.generate_token(prev_tokens)
        return sent

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        i = 0

        if not prev_tokens:
            prev_tokens = tuple()
        assert len(prev_tokens) == n - 1

        probs = self._sorted_probs[tuple(prev_tokens)]
        random_prob = random.random()

        # w word
        # p probabilidad
        w, p = probs[0]
        acc = p

        while random_prob > acc:
            i += 1
            w, p = probs[i]
            acc += p

        return w
