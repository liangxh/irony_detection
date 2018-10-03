# -*- coding: utf-8 -*-
import copy
import random
import numpy as np
from algo.model.const import *


class IndexIterator(object):
    def __init__(self, gold_labels):
        self.gold_labels = np.asarray(gold_labels)
        self._n_sample = self.gold_labels.size
        self.index = np.arange(self._n_sample)
        self.dim = self.gold_labels.max() + 1
        self.label_index = {
            label: self.index[self.gold_labels == label]
            for label in range(self.dim)
        }
        self.mode_index = None

    @property
    def label_n_sample(self):
        return {_label: len(_index) for _label, _index in self.label_index.items()}

    def n_sample(self, mode=None):
        return self._n_sample if mode is None else len(self.mode_index[mode])

    def label_count(self, mode=None):
        if mode is None:
            return {_label: len(_index) for _label, _index in self.label_index.items()}
        else:
            raise NotImplementedError

    def split_train_valid(self, valid_rate):
        self.mode_index = {TRAIN: list(), VALID: list()}
        for label, index in self.label_index.items():
            index = index.tolist()
            random.shuffle(index)
            n_sample = len(index)
            n_valid = int(n_sample * valid_rate)

            self.mode_index[TRAIN] += index[:-n_valid]
            self.mode_index[VALID] += index[-n_valid:]

    def iterate(self, batch_size, mode=None, shuffle=False):
        index = copy.deepcopy(self.mode_index[mode]) if mode is not None else range(self._n_sample)
        if shuffle:
            random.shuffle(index)
        n_patch = batch_size - len(index) % batch_size
        if n_patch < batch_size:
            index += index[:n_patch]
        for n in range(len(index) / batch_size):
            yield index[(n * batch_size): ((n + 1) * batch_size)]


if __name__ == '__main__':
    index_iterator = IndexIterator([0, 0, 0, 0, 0, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1])
    index_iterator.split_train_valid(0.4)

    print 'ALL'
    for batch in index_iterator.iterate(4):
        print batch

    print 'TRAIN'
    for batch in index_iterator.iterate_train(4):
        print batch

    print 'TEST'
    for batch in index_iterator.iterate_valid(4):
        print batch

    index_iterator.split_train_valid(0.4)

    print 'TRAIN'
    for batch in index_iterator.iterate_train(3):
        print batch

    print 'TEST'
    for batch in index_iterator.iterate_valid(3):
        print batch

