# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math
import itertools


def log2ext(x):
    """
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
        prob = 0.0
        for sent in sents:
            log_prob = self.sent_log_prob(sent)
            if log_prob == -math.inf:
                return log_prob
            prob += log_prob
        return prob

    def cross_entropy(self, sents):
        """Cross-entropy of a list of sentences.

        sents -- the sentences.
        """
        prob = self.log_prob(sents)
        m = sum(len(sents) + 1 for sent in sents)
        entropy = float(-prob / m)
        return entropy

    def perplexity(self, sents):
        """Perplexity of a list of sentences.

        sents -- the sentences.
        """
        return math.pow(2.0, self.cross_entropy(sents))


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

        count_prev_tokens = self.count(prev_tokens)

        if count_prev_tokens != 0:
            prob = float(self.count(tokens) / count_prev_tokens)
        return prob

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        n = self._n

        sent = addmarks(sent, n)

        prob = 1
        for i in range(n - 1, len(sent)):
            prob *= self.cond_prob(sent[i], tuple(sent[i - n + 1: i]))
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
            cond_prob = self.cond_prob(sent[i], tuple(sent[i - n + 1: i]))
            # aplico log sobre las probabilidades y las sumo
            prob += log2ext(cond_prob)
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
        # chain.from_iterable equivale a for sent in sents: for elem in sent
        # despliego todos los sents y armo un conjunto agrengando el toke de cierre
        self._voc = voc = set(list(itertools.chain.from_iterable(sents)) + ['</s>'])
        # WORK HERE!!
        self._V = len(voc)  # vocabulary size

    def V(self):
        """Size of the vocabulary."""
        return self._V

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self._n
        V = self._V
        prob = 0

        if not(prev_tokens):
            prev_tokens = tuple()

        # prev_tokens = tuple(prev_tokens) if prev_tokens else tuple()
        assert len(prev_tokens) == n - 1
        # c(wi-s) + V
        count_prev_tokens = self.count(prev_tokens) + V
        tokens = prev_tokens + (token, )

        # c(wi-1, wi)+ 1
        count_tokens = self.count(tokens) + 1

        if count_prev_tokens != 0:
            # p = c(wi-1, wi)+ 1 / c(wi-s) + V
            prob = count_tokens / count_prev_tokens
        return prob


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
        count = defaultdict(int)
        for sent in train_sents:
            count[()] += len(sent) + 1
            for k in range(1, n + 1):
                marked_sent = addmarks(sent, n)
                for i in range(len(marked_sent) - k + 1):
                    ngram = tuple(marked_sent[i: i + k])
                    count[ngram] += 1

        self._count = dict(count)
        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            print('Computing vocabulary...')
            self._voc = voc = set()
            # WORK HERE!!
            # check it
            voc = list(itertools.chain.from_iterable(sents)) + ['</s>']
            self._V = len(voc)

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')
            # WORK HERE!!
            min_gamma, min_p = None, float('inf')
            # use grid search to choose gamma
            for gamma in [1 + i * 5 for i in range(1000)]:
                self._gamma = gamma
                perp = self.perplexity(held_out_sents)
                print("Gamma {} Perplexity {}".format(gamma, perp))

                if perp < min_p:
                    min_gamma, min_p = gamma, perp
            self._gamma = gamma
            print("Choosed gamma {}".format(min_gamma))
            print("With perplexity {}".format(min_p))


    def count(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # init variables
        gamma = self._gamma
        n = self._n
        prob = 0

        
        if prev_tokens is None:
            prev_tokens = ()
        
        # join tokens
        ngram = prev_tokens + (token,)
        
        prev_lambdas_factor = 1.
        
        # calculate q_ML from n-gram down to unigram
        for i in range(n):
            # q_ML(token | prev_tokens) = count_token / count_prev_tokens.
            # we want q_ML for n-grams n={1,2,3,..,n} i.e.
            # q_ML(xn| xi...xn-1) that i={1, 2, 3,...,n-1}.

            i_gram = ngram[i:]
            i_count = self.count(i_gram)
            
            i_less_one_gram = i_gram[:-1] 
            #c(xi...xn−1)
            i_less_one_count = self.count(i_less_one_gram)
            
            # For every i-gram with 1 < k
            if 1 < len(i_gram):
                prob += prev_lambdas_factor * i_count / (i_less_one_count + gamma)
                curr_lamb = prev_lambdas_factor * i_less_one_count / (i_less_one_count + gamma)
                prev_lambdas_factor -= curr_lamb
            # For 1-grams with addone smoothing
            elif len(i_gram) == 1 and self._addone:
                # c(i_gram) +1 / c(i_gram_-1) + V
                prob += prev_lambdas_factor * (i_count + 1) / (i_less_one_count + self._V)
            # For 1-grams without addone
            else:
                prob += prev_lambdas_factor * i_count / i_less_one_count
        
        return prob


class BackOffNGram:
 
    def __init__(self, n, sents, beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.
 
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
 
    """
       Todos los métodos de NGram.
    """
 
    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
 
    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """
 
    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.
 
        tokens -- the k-gram tuple.
        """