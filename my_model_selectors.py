import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def bic_score(self, n):
        """ Utility function to calculate BIC score """
        model = self.base_model(n)
        # log likelihood of the fitted model
        logL = model.score(self.X, self.lengths)
        # N is the number of data points
        logN = np.log(len(self.X))

        # p is the number of parameters
        p = (n ** 2) + 2 * model.n_features * n - 1

        return -2.0 * logL + p * logN

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            # initialization
            best_score = float("inf")
            best_n_components = 1

            for n_components in range(self.min_n_components, self.max_n_components + 1):
                score = self.bic_score(n_components)
                # if new model has better score (The lower the AIC/BIC value the better the model),
                # select that model as candidate best model
                if score < best_score:
                    best_score = score
                    best_n_components = n_components
            return self.base_model(best_n_components)
        except:
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def dic_score(self, n):
        """
            DIC = log(P(X(i)) - 1/(M-1) SUM(log(P(X(all but i))
        """
        model = self.base_model(n)
        # log(P(X(i))
        logL = model.score(self.X, self.lengths)

        scores = []
        for word, (X, lengths) in self.hwords.items():
            # calculate the score of all except i
            if word != self.this_word:
                scores.append(model.score(X, lengths))
        return logL - sum(scores) / len(scores)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            # initialization
            best_score = float("-inf")
            best_n_components = 1

            for n_components in range(self.min_n_components, self.max_n_components + 1):
                score = self.dic_score(n_components)
                # if new model has better DIC score, select that model as candidate best model
                if score > best_score:
                    best_score = score
                    best_n_components = n_components
            return self.base_model(best_n_components)
        except:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def cv_score(self, n):
        """
        Calculate the mean score of cross-validation folds using the KFold class
        :return: the mean Log Likelihood of the model with n number of components
        """
        scores = []
        # split in 5 Kfold, if number of sequences is less than 5, split every sequence
        n_splits = min(5, len(self.sequences))
        split_method = KFold(n_splits=n_splits)

        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
            model = self.base_model(n)
            X, l = combine_sequences(cv_test_idx, self.sequences)
            scores.append(model.score(X, l))
        return np.mean(scores)

    def select(self):
        """ Select the best number of components for the model utilizing cross validation"""
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            # initialization
            best_score = float("-inf")
            best_n_components = self.n_constant

            for n_components in range(self.min_n_components, self.max_n_components + 1):
                score = self.cv_score(n_components)
                # if new model has better score (greater Log Likelihood), select that model as candidate best model
                if score > best_score:
                    best_score = score
                    best_n_components = n_components
            return self.base_model(best_n_components)
        except:
            return self.base_model(self.n_constant)
