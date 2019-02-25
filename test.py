
import pytest

from .vectorizer import Vectorizer
import pandas as pd


df = pd.DataFrame({"Word": ["Hello", "World", "Hello", "World"],
                   "One Hot": ["Hello", "What", "Bye", "Bye"],
                   "Raw": [1, 2, 3, 4]})


class TestShape(object):
    def test_word_vectorizer(self):
        word_vec = Vectorizer(df, word_column="Word").word_encoder()
        assert(word_vec.shape == (4, 2))

    def test_one_hot_vectorizer(self):
        onehot_vec = Vectorizer(df, one_hot_column="One Hot").onehot_encoder()
        assert(onehot_vec.shape == (4, 3))

    def test_raw_vectorizer(self):
        raw_vec = Vectorizer(df, raw_column="Raw").raw_encoder()
        assert(raw_vec.shape == (4, 1))
