# models.py

from optimizers import *
from nerdata import *
from utils import *

from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import logsumexp
import pdb


class ProbabilisticSequenceScorer(object):
    """
    Scoring function for sequence models based on conditional probabilities.
    Scores are provided for three potentials in the model: initial scores (applied to the first tag),
    emissions, and transitions. Note that CRFs typically don't use potentials of the first type.

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """

    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs: np.ndarray, transition_log_probs: np.ndarray, emission_log_probs: np.ndarray):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence_tokens: List[Token], tag_idx: int):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence_tokens: List[Token], prev_tag_idx: int, curr_tag_idx: int):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence_tokens: List[Token], tag_idx: int, word_posn: int):
        word = sentence_tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    """
    HMM NER model for predicting tags

    Attributes:
        tag_indexer: Indexer mapping BIO tags to indices. Useful for dynamic programming
        word_indexer: Indexer mapping words to indices in the emission probabilities matrix
        init_log_probs: [num_tags]-length array containing initial sequence log probabilities
        transition_log_probs: [num_tags, num_tags] matrix containing transition log probabilities (prev, curr)
        emission_log_probs: [num_tags, num_words] matrix containing emission log probabilities (tag, word)
    """

    def __init__(self, tag_indexer: Indexer, word_indexer: Indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def decode(self, sentence_tokens: List[Token]):
        """
        See BadNerModel for an example implementation
        :param sentence_tokens: List of the tokens in the sentence to tag
        :return: The LabeledSentence consisting of predictions over the sentence
        """
        v = self.init_log_probs + self.emission_log_probs[:, self.word_indexer.index_of(sentence_tokens[0].word)]  # [num_tags]
        T = len(sentence_tokens)
        N = len(self.tag_indexer)
        bps = np.zeros((T - 1, N), dtype=np.int)
        for t in range(1, T):
            v_old = v[:, np.newaxis]
            tmp = v_old + self.transition_log_probs
            word = sentence_tokens[t].word
            word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.index_of("UNK")
            tmp = tmp + self.emission_log_probs[:, word_idx][np.newaxis, :]  # [num_tags, num_tags]
            v = np.max(tmp, axis=0)
            bps[t - 1] = np.argmax(tmp, axis=0)

        pred_tags_idx = [np.argmax(v)]
        for t in range(T - 2, -1, -1):
            last = pred_tags_idx[-1]
            pred_tags_idx.append(bps[t, last])
        pred_tags_idx.reverse()
        pred_tags = [self.tag_indexer.get_object(idx) for idx in pred_tags_idx]
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))


def train_hmm_model(sentences: List[LabeledSentence]) -> HmmNerModel:
    """
    Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
    Any word that only appears once in the corpus is replaced with UNK. A small amount
    of additive smoothing is applied.
    :param sentences: training corpus of LabeledSentence objects
    :return: trained HmmNerModel
    """
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.add_and_get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer), len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer), len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.add_and_get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_idx] += 1.0
            else:
                transition_counts[tag_indexer.add_and_get_index(bio_tags[i - 1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print("Tag indexer: %s" % tag_indexer)
    print("Initial state log probabilities: %s" % init_counts)
    print("Transition log probabilities: %s" % transition_counts)
    print("Emission log probs too big to print...")
    print("Emission log probs for India: %s" % emission_counts[:, word_indexer.add_and_get_index("India")])
    print("Emission log probs for Phil: %s" % emission_counts[:, word_indexer.add_and_get_index("Phil")])
    print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


def get_word_index(word_indexer: Indexer, word_counter: Counter, word: str) -> int:
    """
    Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
    At test time, unknown words will be replaced by UNKs.
    :param word_indexer: Indexer mapping words to indices for HMM featurization
    :param word_counter: Counter containing word counts of training set
    :param word: string word
    :return: int of the word index
    """
    if word_counter[word] < 1.5:
        return word_indexer.add_and_get_index("UNK")
    else:
        return word_indexer.add_and_get_index(word)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    def decode(self, sentence_tokens):
        num_words = len(sentence_tokens)
        feature_cache = \
        [
            [
                [] for j in range(0, len(self.tag_indexer))
            ] for i in range(0, len(sentence_tokens))
        ]

        for word_idx in range(0, len(sentence_tokens)):
            for tag_idx in range(0, len(self.tag_indexer)):
                feature_cache[word_idx][tag_idx] = extract_emission_features(
                    sentence_tokens, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer, add_to_indexer=False)

        col = []
        row = []
        num_tags = len(feature_cache[0])
        for word_idx in range(0, num_words):
            for tag_idx in range(0, num_tags):
                row.extend([word_idx * num_tags + tag_idx for i in range(0, len(feature_cache[word_idx][tag_idx]))])
                col.extend(feature_cache[word_idx][tag_idx])
        row = np.array(row)
        col = np.array(col)
        feat_shape = len(self.feature_indexer)
        data = np.ones(len(col), dtype=np.int)
        feats = csr_matrix((data, (row, col)), shape=(num_words * num_tags, feat_shape))
        phi_es = feats.dot(np.expand_dims(self.feature_weights, axis=1)).reshape(num_words, num_tags)

        phi_ts = read_phi_ts()

        v = phi_es[0]
        N = len(self.tag_indexer)
        bps = np.zeros((num_words - 1, N), dtype=np.int)
        for word_idx in range(1, num_words):
            v_old = v[:, np.newaxis]    # [num_tags, 1]
            tmp = v_old + phi_ts    # [num_tags, num_tags]
            tmp = tmp + phi_es[word_idx][np.newaxis, :]     # [num_tags, num_tags]
            v = np.max(tmp, axis=0)
            bps[word_idx - 1] = np.argmax(tmp, axis=0)

        pred_tags_idx = [np.argmax(v)]
        for word_idx in range(num_words - 2, -1, -1):
            last = pred_tags_idx[-1]
            pred_tags_idx.append(bps[word_idx, last])
        pred_tags_idx.reverse()
        pred_tags = [self.tag_indexer.get_object(idx) for idx in pred_tags_idx]
        return LabeledSentence(sentence_tokens, chunks_from_bio_tag_seq(pred_tags))

# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)
    print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = \
        [
            [
                [
                    [] for k in range(0, len(tag_indexer))
                ] for j in range(0, len(sentences[i]))
            ] for i in range(0, len(sentences))
        ]
    for sentence_idx in range(0, len(sentences)):
    # for sentence_idx in range(10):
        if sentence_idx % 1000 == 0:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(
                    sentences[sentence_idx].tokens, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)

    print("use heuristic transition weights")
    phi_ts = read_phi_ts()

    print("Training")
    crf = CrfNerModel(tag_indexer, feature_indexer, np.random.normal(size=len(feature_indexer)))
    # optimizer = Adam(crf.feature_weights)
    optimizer = Adagrad(crf.feature_weights)
    for epoch in range(15):
        sentence_indices = np.arange(len(sentences))
        # sentence_indices = np.arange(10)
        np.random.shuffle(sentence_indices)
        num_iterations = 0
        log_likelihood_sum = 0.
        for sentence_idx in sentence_indices:
            if num_iterations % 5 == 0 or num_iterations % 5 == 1:
                sentence = sentences[sentence_idx]
                labels = sentence.get_bio_tags()
                word_indices = np.arange(len(sentence))
                feats_loc = np.zeros((len(word_indices), len(tag_indexer), len(feature_cache[1][0][0])), dtype=np.int)
                for word_idx in word_indices:
                    for tag_idx in range(0, len(tag_indexer)):
                        feats_loc[word_idx][tag_idx] = feature_cache[sentence_idx][word_idx][tag_idx]

                num_words = feats_loc.shape[0]
                num_tags = feats_loc.shape[1]
                num_feats = feats_loc.shape[2]
                feat_shape = len(feature_indexer)
                col = feats_loc.flatten()
                row = np.tile(np.arange(num_words * num_tags)[:, np.newaxis], (1, num_feats)).flatten()
                data = np.ones(len(col), dtype=np.int)
                feats = csr_matrix((data, (row, col)), shape=(num_words * num_tags, feat_shape))
                phi_es = feats.dot(np.expand_dims(optimizer.weights, axis=1)).reshape(num_words, num_tags)

                alpha, beta, denominator, pyx = forward_backward(phi_es, num_words, num_tags, phi_ts)
                log_likelihood = 0.
                gradient = csr_matrix((1, len(feature_indexer)))
                for word_idx in word_indices:
                    tag = labels[word_idx]
                    tag_idx = tag_indexer.index_of(tag)
                    phi_e = phi_es[word_idx, tag_idx]   # scalar
                    if word_idx > 0:
                        last_tag = labels[word_idx - 1]
                        last_tag_idx = tag_indexer.index_of(last_tag)
                        log_likelihood += phi_ts[last_tag_idx, tag_idx]
                    log_likelihood += phi_e

                    gradient += feats[word_idx * num_tags + tag_idx]
                expect = (feats.transpose().dot(pyx.reshape((num_words * num_tags), 1))).T
                gradient -= csr_matrix(expect)
                optimizer.apply_gradient_update(gradient)
                log_likelihood -= denominator[0]
                log_likelihood_sum += log_likelihood
                if num_iterations % 400 == 0:
                    print("Train Epoch: {} [{}/{} ({:.0f}%)]\tlog likelihood: {:.6f}".format(
                        epoch, num_iterations, len(sentence_indices), 100. * num_iterations / len(sentence_indices),
                        log_likelihood))
            num_iterations += 1
    
        print("====> Epoch: {} Average log likelihood: {:.4f}".format(
            epoch, log_likelihood_sum / len(sentence_indices)))
    
        if (epoch + 1) % 3 == 0:
            print("=> saving checkpoint at epoch {}".format(epoch + 1))
            np.save("model_" + str(epoch + 1), optimizer.weights)
    crf.feature_weights = optimizer.weights
    # crf.feature_weights = np.load("model_10.npy")
    return crf

def read_phi_ts():
    df = pd.read_csv("transition.csv", index_col=0)
    phi_ts = df.to_numpy()
    phi_ts = phi_ts - logsumexp(phi_ts, axis=1)
    return phi_ts

def forward_backward(phi_es, num_words, num_tags, phi_ts):
    alpha = np.zeros((num_words, num_tags))
    alpha[0] = phi_es[0]
    for t in range(1, num_words):
        tmp = alpha[t - 1][:, np.newaxis] + phi_ts  # [num_tags(t-1), num_tags(t)]
        tmp = logsumexp(tmp, axis=0)  # [num_tags(t)]
        alpha[t] = tmp + phi_es[t]  # [num_tags(t)]

    beta = np.zeros((num_words, num_tags))
    for t in range(num_words - 2, -1, -1):
        tmp = beta[t + 1] + phi_es[t + 1]  # [num_tags(t+1)]
        tmp = tmp[np.newaxis, :]    # [num_tags(t), num_tags(t+1)]
        tmp = tmp + phi_ts  # [num_tags(t), num_tags(t+1)]
        beta[t] = logsumexp(tmp, axis=1)   # [num_tags(t)]

    denominator = logsumexp(alpha + beta, axis=1)   # [num_words]
    assert(abs(denominator - denominator[0]).sum() < 1e-4)
    pyx = np.exp(alpha + beta - denominator[:, np.newaxis])     # [num_words, num_tags]
    return alpha, beta, denominator, pyx


def extract_emission_features(sentence_tokens: List[Token], word_index: int, tag: str, feature_indexer: Indexer, add_to_indexer: bool):
    """
    Extracts emission features for tagging the word at word_index with tag.
    :param sentence_tokens: sentence to extract over
    :param word_index: word index to consider
    :param tag: the tag that we're featurizing for
    :param feature_indexer: Indexer over features
    :param add_to_indexer: boolean variable indicating whether we should be expanding the indexer or not. This should
    be True at train time (since we want to learn weights for all features) and False at test time (to avoid creating
    any features we don't have weights for).
    :return: an ndarray
    """
    feats = []
    curr_word = sentence_tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_word = "</s>"
        else:
            active_word = sentence_tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence_tokens):
            active_pos = "</S>"
        else:
            active_pos = sentence_tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size + 1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)
