# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from algo.model.const import *


def basic_evaluate(gold, pred, pos_label=None):
    """
    :param gold: list of int
    :param pred: list of int
    :param pos_label: int, 若指定则precision, recall, f1只返回对应label的值
    :return:
    """
    gold = np.asarray(gold).astype(int)
    pred = np.asarray(pred).astype(int)

    if not gold.shape == pred.shape:
        raise ValueError('gold.shape( {} ) != pred.shape( {} )'.format(gold.shape, pred.shape))

    dim = max(gold.max(), pred.max()) + 1

    matrix = np.zeros((dim, dim))
    for g, p in zip(gold, pred):
        matrix[g][p] += 1

    n_sample = gold.size

    accuracy = float(matrix[range(dim), range(dim)].sum()) / n_sample
    precision_components = list()
    recall_components = list()
    f1_components = list()

    for label in range(dim):
        _n_pred = matrix[:, label].sum()
        _n_gold = matrix[label, :].sum()
        _n_correct = matrix[label][label]

        _precision = float(_n_correct) / _n_pred if _n_pred != 0. else 0.
        _recall = float(_n_correct) / _n_gold if _n_gold != 0. else 0.
        _f1 = 2. * (_precision * _recall) / (_precision + _recall) if (_precision + _recall) != 0 else 0.

        precision_components.append(_precision)
        recall_components.append(_recall)
        f1_components.append(_f1)

    if pos_label is not None:
        precision = precision_components[pos_label]
        recall = recall_components[pos_label]
        f1 = f1_components[pos_label]
    else:
        # macro-average
        precision = sum(precision_components) / dim
        recall = sum(recall_components) / dim
        f1 = sum(f1_components) / dim

    return {
        ACCURACY: accuracy,
        PRECISION: precision, RECALL: recall, F1_SCORE: f1,
        PRECISION_COMPONENTS: precision_components,
        RECALL_COMPONENTS: recall_components,
        F1_SCORE_COMPONENTS: f1_components,
        CONFUSION_MATRIX: matrix.tolist()
    }


if __name__ == '__main__':
    gold = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    pred = [2, 0, 0, 1, 1, 0, 1, 2, 2]
    res = basic_evaluate(gold, pred)
    print(res)
    print()

    res = basic_evaluate(gold, pred, pos_label=0)
    print(res)
