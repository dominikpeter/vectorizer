
import pandas as pd
import numpy as np

from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize


def val_to_list(val):
    if isinstance(val, (tuple, list)):
        return val
    else:
        assert(isinstance(val, str))
        return [val]


def check_column_in_df(df, cols):
    for i in cols:
        if i not in df.columns:
            return False
    return True


def check_dtype_is_number(train):
    return np.issubdtype(train, np.number)


treebank_word_tokenizer = TreebankWordTokenizer().tokenize


class Vectorizer(object):
    """Vecotrizing features
    Alwatests returning a sparse matritrain

    Parameters
    ----------
    train :             A DataFrame.
    test :              A DataFrame, the result will have the same number
                        of columns as the first train result default=None.
    one_hot_column :    Column name(s) to turn into one hot columns, default=None.
    word_column :       Column name(s) to turn into word counts, default=None
    raw_column :        Column name(s) to return as sparse matritrain with optional transformation
                        default=None
    word_tokenizer :    Tokenizer for all columns or a dict with a tokenizer for each column
                        default=treebank_word_tokenizer
    word_vectorizer :   WordVectorizer for all columns or a dict with a tokenizer for each column
                        default=TfidfVectorizer

    Attributes
    ----------
    encoded_train_ :    Spare matritrain with the encoded columns
    encoded_test_ :     Sparse matritrain with the encoded columns

    Examples
    --------
    df = pd.DataFrame({"Word": ["Hello", "World", "Hello", "World"],
                       "One Hot": ["Hello", "What", "bye", "bye"],
                       "Raw": [1, 2, 3, 4]})

    Vectorizer(df, word_column="Word").word_encode()
    Vectorizer(df, one_hot_column="One Hot").onehot_encode()
    Vectorizer(df, raw_column="Raw").raw_encode()
    Vectorizer(df, word_column="Word",
               one_hot_column="One Hot",
               raw_column="Raw").encode()
    """

    def __init__(self, train, test=None,
                 one_hot_column=None,
                 word_column=None,
                 raw_column=None,
                 word_tokenizer=treebank_word_tokenizer,
                 word_vectorizer=TfidfVectorizer,
                 normalize=True):
        assert(isinstance(train, pd.DataFrame))
        assert(any([one_hot_column, word_column, raw_column]))
        self._train = train.copy()
        if test is not None:
            assert(isinstance(test, pd.DataFrame))
            self._test = test.copy()
        else:
            self._test = None

        self._has_word_column = True if word_column else False
        self._has_onehot_column = True if one_hot_column else False
        self._has_raw_column = True if raw_column else False

        if self._has_word_column:
            self._word_column = val_to_list(word_column)
            self._check_column_in_df(self._word_column)
        if self._has_raw_column:
            self._raw_column = val_to_list(raw_column)
            self._check_column_in_df(self._raw_column)
        if self._has_onehot_column:
            self._one_hot_column = val_to_list(one_hot_column)
            self._check_column_in_df(self._one_hot_column)

        if self._has_word_column:
            self._word_tokenizer = self._word_transformer_to_dict(
                self._word_column, word_tokenizer)
            self._word_vecotrizer = self._word_transformer_to_dict(
                self._word_column, word_vectorizer)

        self._normalize = normalize

    def _check_column_in_df(self, to_check):
        assert(check_column_in_df(self._train, to_check))
        if self._test is not None:
            assert(check_column_in_df(self._test, to_check))

    def _hstack(self, fun, columns):
        if self._test is not None:
            train_stack = hstack([fun(i)[0] for i in columns])
            test_stack = hstack([fun(i)[1] for i in columns])
            return train_stack, test_stack
        else:
            train_stack = hstack([fun(i) for i in columns])
        return train_stack

    def _onehot_encode(self, column):
        onehot = OneHotEncoder(handle_unknown="ignore")
        onehot.fit(self._train[column].values.reshape(-1, 1))
        onehot_train = onehot.transform(
            self._train[column].values.reshape(-1, 1))
        if self._test is not None:
            onehot_test = onehot.transform(
                self._test[column].values.reshape(-1, 1))
            return onehot_train, onehot_test
        else:
            return onehot_train

    def onehot_encode(self):
        assert(self._has_onehot_column)
        return self._hstack(self._onehot_encode, self._one_hot_column)

    def _word_transformer_to_dict(self, val, transformer):
        if isinstance(val, dict):
            return val
        else:
            return {v: transformer for v in val}

    def _word_encode(self, column):
        tokenizer_ = self._word_tokenizer[column]
        vectorizer_ = self._word_vecotrizer[column]
        wordv = vectorizer_(tokenizer=tokenizer_)
        wordv.fit(self._train[column])
        word_vector_train = wordv.transform(self._train[column])
        if self._test is not None:
            word_vector_test = wordv.transform(self._test[column])
            return word_vector_train, word_vector_test
        return word_vector_train

    def word_encode(self):
        assert(self._has_word_column)
        return self._hstack(self._word_encode, self._word_column)

    def raw_encode(self):
        assert(self._has_raw_column)
        assert(
            all([check_dtype_is_number(self._train[i]) for i in self._raw_column]))
        if self._test is not None:
            assert(all([check_dtype_is_number(
                        self._test[i]) for i in self._raw_column]))
        return self._raw_encode()

    def _raw_encode(self):
        raw_train = hstack(
            [csr_matrix(self._train[i][:, np.newaxis]) for i in self._raw_column])
        if self._test is not None:
            raw_test = hstack(
                [csr_matrix(
                    self._test[i][:, np.newaxis]) for i in self._raw_column])
        if self._normalize:
            raw_train = normalize(raw_train)
            if self._test is not None:
                raw_test = normalize(raw_test)
        if self._test is not None:
            return raw_train, raw_test
        return raw_train

    def _encode(self, *encoders):
        encoded_vectors = [encoder() for encoder in encoders]
        if self._test is not None:
            self.encoded_train_ = hstack([i[0] for i in encoded_vectors])
            self.encoded_test_ = hstack([i[1] for i in encoded_vectors])
            return self.encoded_train_, self.encoded_test_
        self.encoded_train_ = hstack([i for i in encoded_vectors])
        return self.encoded_train_

    def encode(self):
        encoders = []
        if self._has_raw_column:
            encoders.append(self.raw_encode)
        if self._has_word_column:
            encoders.append(self.word_encode)
        if self._has_onehot_column:
            encoders.append(self.onehot_encode)
        return self._encode(*encoders)
