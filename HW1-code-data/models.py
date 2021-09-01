# models.py

from sentiment_data import *
from utils import *
import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy.sparse import csr_matrix

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, ex_words: List[str], add_to_indexer: bool=False) -> List[int]:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param ex_words: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return:
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer, train_exs, stop_words):
        for sentimentExample in train_exs:
            words = sentimentExample.words
            for word in words:
                lowercase = word.lower()
                if not lowercase in stop_words:
                    indexer.add_and_get_index(lowercase)
        self.indexer = indexer
        self.corpus_length = len(indexer)

        self.feats = []
        for i, sentimentExample in enumerate(train_exs):
            sentence = sentimentExample.words
            self.feats.append(self.calculate_sentence_probability(sentence))

    def calculate_sentence_probability(self, sentence):
        col = [self.indexer.index_of(word.lower()) for word in sentence if self.indexer.contains(word.lower())]
        row = np.zeros(len(col), dtype=np.int)
        data = np.ones(len(col), dtype=np.int)
        feat = csr_matrix((data, (row, col)), shape=(1, self.corpus_length))
        if len(col) > 0:
            feat = feat * (1. / len(col))
        return feat

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer, train_exs, stop_words):
        for sentimentExample in train_exs:
            words = sentimentExample.words
            previous_word = None
            for word in words:
                if previous_word is not None:
                    if not (previous_word.lower() in stop_words and word.lower() in stop_words):
                        indexer.add_and_get_index((previous_word.lower(), word.lower()))
                previous_word = word
        self.indexer = indexer
        self.corpus_length = len(indexer)

        self.feats = []
        for i, sentimentExample in enumerate(train_exs):
            sentence = sentimentExample.words
            self.feats.append(self.calculate_sentence_probability(sentence))

    def calculate_sentence_probability(self, sentence):
        col = []
        previous_word = None
        for word in sentence:
            if previous_word is not None:
                if self.indexer.contains((previous_word.lower(), word.lower())):
                    col.append(self.indexer.index_of((previous_word.lower(), word.lower())))
            previous_word = word
        row = np.zeros(len(col), dtype=np.int)
        data = np.ones(len(col), dtype=np.int)
        feat = csr_matrix((data, (row, col)), shape=(1, self.corpus_length))
        if len(col) > 0:
            feat = feat * (1. / len(col))
        return feat

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer, train_exs, stop_words):
        # unigram
        for sentimentExample in train_exs:
            words = sentimentExample.words
            for word in words:
                lowercase = word.lower()
                if not lowercase in stop_words:
                    indexer.add_and_get_index(lowercase)

        # bigram
        for sentimentExample in train_exs:
            words = sentimentExample.words
            previous_word = None
            for word in words:
                if previous_word is not None:
                    if not (previous_word.lower() in stop_words and word.lower() in stop_words):
                        indexer.add_and_get_index((previous_word.lower(), word.lower()))
                previous_word = word

        self.indexer = indexer
        self.corpus_length = len(indexer)

        self.feats = []
        for i, sentimentExample in enumerate(train_exs):
            sentence = sentimentExample.words
            self.feats.append(self.calculate_sentence_probability(sentence))

    def calculate_sentence_probability(self, sentence):
        col = [self.indexer.index_of(word.lower()) for word in sentence if self.indexer.contains(word.lower())]
        unigram_count = len(col)

        previous_word = None
        for word in sentence:
            if previous_word is not None:
                if self.indexer.contains((previous_word.lower(), word.lower())):
                    col.append(self.indexer.index_of((previous_word.lower(), word.lower())))
            previous_word = word
        bigram_count = len(col) - unigram_count
        row = np.zeros(len(col), dtype=np.int)
        data = np.ones(len(col))
        data[:unigram_count] = data[:unigram_count] * 1. / unigram_count
        data[unigram_count:unigram_count + bigram_count] = data[unigram_count:unigram_count + bigram_count] * 1. / bigram_count
        feat = csr_matrix((data, (row, col)), shape=(1, self.corpus_length))
        return feat

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex_words: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, ex_words: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self):
        raise Exception("Must be implemented")


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_size, feat_extractor):
        self.w = np.zeros(feat_size)
        self.feat_extractor = feat_extractor

    def predict(self, sentence):
        feat = self.feat_extractor.calculate_sentence_probability(sentence)
        return int(feat.dot(np.expand_dims(self.w, axis=1))[0, 0] > 0)

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    raise Exception("Must be implemented")

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    lr = LogisticRegressionClassifier(feat_extractor.corpus_length, feat_extractor)
    alpha = 1e1
    # beta = 1e-4
    for epoch in range(4):
        loss = 0.
        acc = 0
        indices = np.arange(len(train_exs))
        np.random.shuffle(indices)
        for i in indices:
            feat = feat_extractor.feats[i]
            sentimentExample = train_exs[i]
            y = sentimentExample.label
            z = 1 / (1 + np.exp(-feat.dot(np.expand_dims(lr.w, axis=1))))[0, 0]
            loss += -y * np.log(z) - (1 - y) * np.log(1 - z) \
                    # + beta * np.expand_dims(lr.w, axis=0).dot(np.expand_dims(lr.w, axis=1))[0, 0]
            predict = int(feat.dot(np.expand_dims(lr.w, axis=1))[0, 0] > 0)
            acc += (predict == y)
            grad = (z - y) * feat.toarray()[0] # + 2 * beta * lr.w
            lr.w = lr.w - alpha * grad
        print("epoch {:d}, loss: {:f}, accuracy: {:f}".format(epoch, loss / len(train_exs), acc / len(train_exs)))

    return lr

def train_model(args, train_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        feat_extractor = UnigramFeatureExtractor(Indexer(), train_exs, stop_words)
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer(), train_exs, stop_words)
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer(), train_exs, stop_words)
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model