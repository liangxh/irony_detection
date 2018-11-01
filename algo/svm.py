# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import importlib
import yaml
import numpy as np
from sklearn import svm
from scipy.sparse import hstack
from dataset.common.const import *
from dataset.common.load import *
from algo.model.const import *
from algo.lib.evaluate import basic_evaluate
from algo.lib.common import print_evaluation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from nlp.lexicon_feat import extract_tf

FEATS = 'feats'
TF_IDF = 'tf-idf'
TF = 'tf'
TF_C = 'tf_c'


class Config(object):
    def __init__(self, data):
        self.data = data

    @property
    def feat_keys(self):
        return self.data['feat_keys']

    @property
    def use_class_weights(self):
        return self.data['train']['use_class_weights']


def normalize_matrix(m):
    return m
    m -= m.mean(axis=0)
    #return m / m.std(axis=0)
    return m
    return m / m.max(axis=0)
    #return (m - m.mean(axis=0)) #/ np.sqrt(m.power(2).sum(axis=0))


def load_dataset(data_config, train_config, label_version=None):
    datasets = dict()
    for mode in [TRAIN, TEST]:
        feats = list()
        for key in train_config.feat_keys:
            path = data_config.output_path(key, mode, HIDDEN_FEAT)
            feat_list, _ = load_feat(path)
            feats.append(feat_list)
        feats = np.concatenate(feats, axis=1) if len(feats) > 1 else np.asarray(feats[0])

        label_path = data_config.path(mode, LABEL, label_version)
        labels_gold = load_label_list(label_path)

        datasets[mode] = {
            LABEL_GOLD: np.asarray(labels_gold),
            FEATS: feats
        }
    return datasets


@commandr.command
def main(dataset_key, label_version=None, config_path='config_svm.yaml'):
    """
    python algo/svm.py main semeval2018_task3 A

    :param dataset_key:
    :param label_version:
    :param config_path:
    :return:
    """
    pos_label = None
    if dataset_key == 'semeval2018_task3' and label_version == 'A':
        pos_label = 1

    config_data = yaml.load(open(config_path))
    train_config = Config(data=config_data)

    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')
    datasets = load_dataset(data_config, train_config, label_version)

    if train_config.use_class_weights:
        class_weight = 'balanced'
    else:
        class_weight = None

    #tf = extract_tf(dataset_key, text_version='ek')

    clf = svm.SVC(class_weight=class_weight)
    #X = hstack([datasets[TRAIN][FEATS], tf[TRAIN]])
    X = datasets[TRAIN][FEATS]
    X = normalize_matrix(X)
    clf.fit(X=X, y=datasets[TRAIN][LABEL_GOLD])

    for mode in [TRAIN, TEST]:
        #X = hstack([datasets[mode][FEATS], tf[mode]])
        X = datasets[mode][FEATS]
        labels_predict = clf.predict(X=X)
        labels_gold = datasets[mode][LABEL_GOLD]
        res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)

        print(mode)
        print_evaluation(res)
        print()


@commandr.command
def main2(dataset_key, label_version=None, config_path='config_svm.yaml'):
    """
    python algo/svm.py main semeval2018_task3 A

    :param dataset_key:
    :param label_version:
    :param config_path:
    :return:
    """
    pos_label = None
    if dataset_key == 'semeval2018_task3' and label_version == 'A':
        pos_label = 1

    config_data = yaml.load(open(config_path))
    train_config = Config(data=config_data)

    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')
    datasets = load_dataset(data_config, train_config, label_version)

    if train_config.use_class_weights:
        class_weight = 'balanced'
    else:
        class_weight = None

    tf = extract_tf(dataset_key, text_version='ek')

    clf = svm.SVC(class_weight=class_weight)
    X = hstack([datasets[TRAIN][FEATS], tf[TRAIN]])
    clf.fit(X=X, y=datasets[TRAIN][LABEL_GOLD])

    for mode in [TRAIN, TEST]:
        X = hstack([datasets[mode][FEATS], tf[mode]])
        labels_predict = clf.predict(X=X)
        labels_gold = datasets[mode][LABEL_GOLD]
        res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)

        print(mode)
        print_evaluation(res)
        print()


@commandr.command
def tf_idf(dataset_key, text_version, label_version=None, use_class_weights=True):
    """
    python algo/svm.py tf_idf semeval2018_task3 -t ek -l A
    """
    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')
    pos_label = None
    if dataset_key == 'semeval2018_task3' and label_version == 'A':
        pos_label = 1

    datasets = dict()
    for mode in [TRAIN, TEST]:
        datasets[mode] = {
            TEXT: load_text_list(data_config.path(mode, TEXT, text_version)),
            LABEL: load_label_list(data_config.path(mode, LABEL, label_version))
        }

    vectorizers = {
        #TF_IDF: TfidfVectorizer(ngram_range=(1, 3), min_df=0.01),
        TF: CountVectorizer(ngram_range=(1, 3), min_df=0.02),
    }

    for key, vectorizer in vectorizers.items():
        feat = vectorizer.fit_transform(datasets[TRAIN][TEXT])
        datasets[TRAIN][key] = feat

        feat = vectorizer.transform(datasets[TEST][TEXT])
        datasets[TEST][key] = feat

    if use_class_weights:
        class_weight = 'balanced'
    else:
        class_weight = None

    clf = svm.SVC(class_weight=class_weight)
    X = hstack([datasets[TRAIN][k] for k in vectorizers.keys()])
    X = normalize_matrix(X)
    clf.fit(X=X, y=datasets[TRAIN][LABEL])

    for mode in [TRAIN, TEST]:
        X = hstack([datasets[mode][k] for k in vectorizers.keys()])
        X = normalize_matrix(X)
        labels_predict = clf.predict(X=X)
        labels_gold = datasets[mode][LABEL]
        res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)

        print(mode)
        print_evaluation(res)
        print()


if __name__ == '__main__':
    commandr.Run()
