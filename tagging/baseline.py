from collections import defaultdict


class BadBaselineTagger:

    def __init__(self, tagged_sents):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        """
        pass

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        return 'nc0s000'

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return True


class BaselineTagger:

    def __init__(self, tagged_sents, default_tag='nc0s000'):
        """
        tagged_sents -- training sentences, each one being a list of pairs.
        default_tag -- tag for unknown words.
        """
        # WORK HERE!!
        self._default_tag = default_tag
        # Build a dict with the following shape
        # {word1: {tag1: amount1, tag2: amount2...} ... wordn: {tag1:amount1, tag2: amount2..}}
        self._word_tag_dict = defaultdict(lambda: defaultdict(int))

        for sent in tagged_sents:
            for word, tag in sent:
                self._word_tag_dict[word][tag] += 1

        self._word_tag_dict = {w: dict(t) for w, t in self._word_tag_dict.items()}

    def tag(self, sent):
        """Tag a sentence.

        sent -- the sentence.
        """
        return [self.tag_word(w) for w in sent]

    def tag_word(self, w):
        """Tag a word.

        w -- the word.
        """
        if self.unknown(w):
            return self._default_tag
        # most frequent tag for w word
        tag = sorted(self._word_tag_dict[w].items(), key=lambda x: x[1],
                     reverse=True)[0][0]
        return tag

    def unknown(self, w):
        """Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self._word_tag_dict.keys()
