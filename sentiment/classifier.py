"""Classifiers."""
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, ParameterGrid


classifiers = {
    'maxent': LogisticRegression,
    'mnb': MultinomialNB,
    'svm': LinearSVC,
}


class CustomTokenizer(object):
    """
    Define custom tokenizer.

    It includes cleaning tweets, tweet tokenizer, replace repeated character
    sequences of length 3 or greater with sequences of length 3
    remove stop words (not remove 'no' word) and lemmatization.
    """

    def __init__(self):
        """Init."""
        self.stop_words = set(stopwords.words('spanish')) - set(['no'])
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, tweets):
        """Call method."""
        return [self.lemmatizer.lemmatize(tweet) for tweet in self.my_tokenizer(self.clean_tweet(tweets))]

    def my_tokenizer(self, tweet):
        """tokenize."""
        tokenizer = TweetTokenizer(reduce_len=True)
        tokens = tokenizer.tokenize(tweet)
        tokens = self.clean_stopwords(tokens)
        return tokens

    def clean_tweet(self, tweet):
        """
        Tweet Normalize fuction.

        Remove user mentions @<user-tag>
        Remove URLS
        """
        pattern = "(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)"
        replace = "@USER"
        normed_content = ' '.join(re.sub(pattern, replace, tweet).split())

        pattern_url = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        replace = "URL"
        cleaned_content = ' '.join(
            re.sub(pattern_url, replace, normed_content).split())
        return cleaned_content

    def clean_stopwords(self, tokens):
        """Remove stop words."""
        return [token for token in tokens if token.lower() not in self.stop_words]


class SentimentClassifier(object):
    """Sentimente Classifier."""

    def __init__(self, clf='svm'):
        """
        Init.

        clf -- classifying model, one of 'svm', 'maxent', 'mnb' (default: 'svm').
        """
        self._clf = clf
        if clf == 'svm':
            classifier = classifiers[clf](random_state=0, C=.1)
        elif clf == 'maxent':
            classifier = classifiers[clf](random_state=0, C=1., penalty='l2')
        elif clf == 'mnb':
            classifier = classifiers[clf]()

        self._pipeline = pipeline = Pipeline([
            # Accuracy: 57.31% (290/506)
            ('vect', TfidfVectorizer(tokenizer=CustomTokenizer(),
                                     max_df=.95,
                                     binary=True, strip_accents='ascii',
                                     min_df=2)),
            ('clf', classifier),
        ])

    def fit(self, X, y):
        """Fit method."""
        self._pipeline.fit(X, y)

    def predict(self, X):
        """Predict method."""
        return self._pipeline.predict(X)

    def cross_validation(self, X, y):
        """Cross validation."""
        param_grid = {
            # 'vect__tokenizer': [TweetTokenizer(reduce_len=True).tokenize],
            # 'vect__binary': [True, False],
            # 'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
            # 'vect__min_df': [1, 3, 5, 7, 9],
            # 'vect__max_df': [.95, .9, .85, .7],
            # 'vect__lowercase': [True, False],
            # 'vect__strip_accents': ['unicode', 'ascii'],
            'clf__random_state': [0],
        }
        if self._clf == 'svm':
            param_grid['clf__C'] = [.1, .01, .001, .00001, 1., 5., 10., 100.]
        elif self._clf == 'maxent':
            param_grid['clf__C'] = [.1, .01, .001, .00001, 1., 5., 10., 100.]
            param_grid['clf__penalty'] = ['l1', 'l2']

        clf = GridSearchCV(self._pipeline, [param_grid], cv=5,
                           scoring='accuracy', verbose=3)
        clf.fit(X, y)

        print("Best parameters set found:")
        print(clf.best_params_)

        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
