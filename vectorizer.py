
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


def check_if_number(dtype):
    return np.issubdtype(dtype, np.number)


def check_dtype_is_number(x):
    return check_if_number(x)


class Vectorizer(object):
    """Vecotrizing features
    Always returning a sparse matrix

    Parameters
    ----------
    categories : 'auto' or a list of lists/arrays of values, default='auto'.
    sparse : boolean, default=True
        Will return sparse matrix if set True else will return an array.
    dtype : number type, default=np.float
        Desired dtype of output.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``).

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.

    """

    def __init__(self, X, y=None,
                 one_hot_column=None,
                 word_column=None,
                 raw_column=None,
                 word_tokenizer=None,
                 word_vectorizer=None,
                 normalize=True):
        assert(isinstance(X, pd.DataFrame))
        assert(any([one_hot_column, word_column, raw_column]))
        self._X = X.copy()
        if y is not None:
            assert(isinstance(y, pd.DataFrame))
            self._y = y.copy()
        else:
            self._y = None

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
                self._word_column, TreebankWordTokenizer().tokenize)
            self._word_vecotrizer = self._word_transformer_to_dict(
                self._word_column, TfidfVectorizer)

        self._normalize = normalize

    def _check_column_in_df(self, to_check):
        assert(check_column_in_df(self._X, to_check))
        if self._y is not None:
            assert(check_column_in_df(self._y, to_check))

    def _word_transformer_to_dict(self, val, transformer):
        if isinstance(val, dict):
            return val
        else:
            return {v: transformer for v in val}

    def _onehot_encoder(self, column):
        onehot = OneHotEncoder(handle_unknown="ignore")
        onehot.fit(self._X[column].values.reshape(-1, 1))
        onehot_X = onehot.transform(
            self._X[column].values.reshape(-1, 1))
        if self._y is not None:
            onehot_y = onehot.transform(
                self._y[column].values.reshape(-1, 1))
            return onehot_X, onehot_y
        else:
            return onehot_X

    def onehot_encoder(self):
        assert(self._has_onehot_column)
        return self._hstack(self._onehot_encoder, self._one_hot_column)

    def word_encoder(self):
        assert(self._has_word_column)
        return self._hstack(self._word_encoder, self._word_column)

    def raw_encoder(self):
        assert(self._has_raw_column)
        assert(
            all([check_dtype_is_number(self._X[i]) for i in self._raw_column]))
        if self._y is not None:
            assert(
                all([check_dtype_is_number(self._y[i]) for i in self._raw_column]))
        return self._raw_encoder()

    def _raw_encoder(self):
        raw_X = hstack(
            [csr_matrix(self._X[i][:,np.newaxis]) for i in self._raw_column])
        if self._y is not None:
            raw_y = hstack(
                [csr_matrix(self._y[i][:,np.newaxis]) for i in self._raw_column])
        if self._normalize:
            raw_X = normalize(raw_X)
            if self._y is not None:
                raw_y = normalize(raw_y)
        if self._y is not None:
            return raw_X, raw_y
        return raw_X

    def _word_encoder(self, column):
        tokenizer_ = self._word_tokenizer[column]
        vectorizer_ = self._word_vecotrizer[column]
        wordv = vectorizer_(tokenizer=tokenizer_)
        wordv.fit(self._X[column])
        word_vector_X = wordv.transform(self._X[column])
        if self._y is not None:
            word_vector_y = wordv.transform(self._y[column])
            return word_vector_X, word_vector_y
        return word_vector_X

    def _hstack(self, fun, columns):
        if self._y is not None:
            X_stack = hstack([fun(i)[0] for i in columns])
            Y_stack = hstack([fun(i)[1] for i in columns])
            return X_stack, Y_stack
        else:
            X_stack = hstack([fun(i) for i in columns])
        return X_stack

    def encode(self):
        encoders = []
        if self._has_raw_column:
            encoders.append(self.raw_encoder)
        if self._has_word_column:
            encoders.append(self.word_encoder)
        if self._has_onehot_column:
            encoders.append(self.onehot_encoder)
        return self._hstack_encoder(*encoders)

    def _hstack_encoder(self, *encoders):
        encoded_vectors = [encoder() for encoder in encoders]
        if self._y is not None:
            self.encoded_vector_X = hstack([i[0] for i in encoded_vectors])
            self.encoded_vector_y = hstack([i[1] for i in encoded_vectors])
            return self.encoded_vector_X, self.encoded_vector_y
        self.encoded_vector_X = hstack([i for i in encoded_vectors])
        return self.encoded_vector_X
