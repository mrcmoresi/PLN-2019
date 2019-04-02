# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math
import itertools


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
        """Init.

        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        # call superclass to compute counts
        super().__init__(n, sents)

        # compute vocabulary
        # chain.from_iterable is the same for sent in sents: for elem in sent
        # expand all the sent and build a set
        self._voc = voc = set(
            list(itertools.chain.from_iterable(sents)) + ['</s>'])
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
    """Intepolate model."""

    def __init__(self, n, sents, gamma=None, addone=True):
        """Init.

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
            for gamma in [1 + i * 5 for i in range(100)]:
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

        # constraint
        # sum(lambdas) = 1 with lambda_i >= 0
        lambdas_factor = 1.

        # calculate q_ML from n-gram down to unigram
        for i in range(n):
            # q_ML(token | prev_tokens) = count_token / count_prev_tokens.
            # we want q_ML for n-grams n={1,2,3,..,n} i.e.
            # q_ML(xn| xi...xn-1) that i={1, 2, 3,...,n-1}.

            # ngram
            i_gram = ngram[i:]
            i_count = self.count(i_gram)

            # ngram-1 with one token less
            i_less_one_gram = i_gram[:-1]
            i_less_one_count = self.count(i_less_one_gram)

            # For every i-gram with 1 < i
            if 1 < len(i_gram):
                prob += lambdas_factor * i_count / (i_less_one_count + gamma)
                curr_lamb = (
                    lambdas_factor * i_less_one_count / (i_less_one_count + gamma))
                lambdas_factor -= curr_lamb
            # For 1-grams with addone
            elif len(i_gram) == 1 and self._addone:
                # c(i_gram) +1 / c(i_gram_-1) + V
                prob += lambdas_factor * (i_count + 1) / (i_less_one_count + self._V)
            # For 1-grams w/o addone
            else:
                prob += lambdas_factor * i_count / i_less_one_count

        return prob


class BackOffNGram(LanguageModel):
    """Back-off NGram model."""

    def __init__(self, n, sents, beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.

        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        # super().__init__(n, sents)
        self.beta = beta
        self.addone = addone
        self.set_A = set_A = defaultdict(set)
        self.voc = voc = set(
            list(itertools.chain.from_iterable(sents)) + ['</s>'])
        self.V = len(voc)
        self.models = models = []
        self.n = n

        if beta is None:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]
        else:
            train_sents = sents

        # unigrams
        if self.addone:
            models.append(AddOneNGram(1, train_sents))
        else:
            models.append(NGram(1, train_sents))
        #  ngrams models n = 2, 3, ... n
        # and put it in a list [1-gram, 2-gram, 3-gram, ... ,n-gram]
        for i in range(2, n + 1):
            models.append(NGram(i, train_sents))

        for model in models[1:]:
            n_of_model = model._n
            for ngram, val in model._count.items():

                # print("ngram {} val {}".format(ngram, val))
                if len(ngram) == n_of_model:
                    n_m1_gram = ngram[:-1]
                    # print(n_m1_gram)

                    # build A set
                    set_A[n_m1_gram].add(ngram[-1])
                    # print(set_A)

        if beta is None:
            # m = int(0.9 * len(sents))
            # held_out_sents = sents[m:]
            beta = self.calculate_beta(held_out_sents)
        else:
            self._alpha = self.calculate_alpha()
            self._denom = self.calculate_denominator()

    def calculate_alpha(self):
        """
        Calculate alpha.

        alpha(x1, ... xi )= 1 - sum(c_discount(x1 .. xi x) / c(x1 .. xi))
        """
        _alpha = defaultdict(float)
        for ngram, amount in self.set_A.items():
            acc = 0.
            # print(self.set_A.items())
            for x in amount:
                # c_discount(x1 .. xi x)
                c_discount = self.count(ngram + tuple([x])) - self.beta
                # c(x1, ..., xi)
                c = self.count(ngram)
                acc += c_discount / c
            _alpha[ngram] = 1. - acc

        return _alpha

    def calculate_denominator(self):
        """Calculate denominator.

        denom(x1, .. xi) = 1 - sum(qd(x|x2 .. xi-1))
        """
        _denom = defaultdict(float)
        for ngram, amount in self.set_A.items():
            # print('Adentro del for')
            acc = 0.
            #print(ngram, amount)
            for x in amount:
                #print(amount)
                #print(x)
                # print(x, ngram[1:])
                acc += self.cond_prob(x, ngram[1:])
            _denom[ngram] = 1. - acc
        return _denom

    def calculate_beta(self, held_out_sents):
        """Estimate beta param using held-out data."""
        print("Begin calculate beta \n")
        temp = 0.
        max_bound = float('-inf')
        # i = [.0 ,.05 ,.1, .15 ... 1]
        for i in [float(x * 0.05) for x in range(21)]:
            self.beta = i
            # need calculate alpha and denominator to calculare log_prob
            self._alpha = self.calculate_alpha()
            self._denom = self.calculate_denominator()
            lp = self.log_prob(held_out_sents)
            print(
                "Beta {} Alpha {} denom {} log prob {}".format(
                    self.beta, self._alpha, self.denom, lp))
            if max_bound < lp:
                max_bound = lp
                temp = i

        # beta between 0 and 1
        assert(temp >= 0 and temp <= 1)
        self.beta = temp
        # recalculate alpha and denominator after new beta
        self._alpha = self.calculate_alpha()
        self._denom = self.calculate_denominator()

    def A(self, tokens):
        """Set of words with counts > 0 for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        return self.set_A.get(tokens, set())

    def alpha(self, tokens):
        """Missing probability mass for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        return self._alpha.get(tokens, 1.)

    def denom(self, tokens):
        """Normalization factor for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        return self._denom.get(tokens, 1.)

    def count(self, tokens):
        """Return length of the token."""
        n = len(tokens)
        # check if it is first token
        if tokens == n * ('<s>',):
            n += 1
        # print("COUUUUNNTT")
        # print(tokens)
        # print(n)
        # print(self.models)
        # print(self.models[n - 1]._count)
        count = self.models[n - 1].count(tokens)
        return count

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        prob = 0.

        # 1 gram model
        if not prev_tokens:
            # print("PREV TOKENS is NONE")
            count = self.count(tuple([token]))
            count_total = self.count(tuple())

            if self.addone:
                prob = (count + 1) / (count_total + self.V)
            else:
                prob = count / count_total
        # all the others n-gram model
        else:
            # check if xi belongs A(x1 , ... , xi-1)
            if token in self.A(tuple(prev_tokens)):
                tokens = tuple(prev_tokens) + tuple([token])
                # c*(x1, ..., xi)
                discounted = self.count(tokens) - self.beta
                # c(x1, ..., xi-1)
                count = self.count(tuple(prev_tokens))
                prob = discounted / float(count)
            else:
                # alpha(x1, ..., xi-1)
                alpha = self.alpha(tuple(prev_tokens))
                # qd(xi|x2 ... xi-1)
                prob_previous = self.cond_prob(token, prev_tokens[1:])
                # denom(x1, ..., xi-1)
                denominator = self.denom(tuple(prev_tokens))
                if prob_previous != 0 and denominator != 0:
                    prob = alpha * (prob_previous / denominator)
        return prob
