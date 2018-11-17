# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import importlib
import yaml
import math
import numpy as np
from sklearn import svm
from scipy.sparse import hstack
from dataset.common.const import *
from dataset.common.load import *
from algo.model.const import *
from algo.lib.evaluate import basic_evaluate
from algo.lib.common import print_evaluation
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

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


def load_dataset(data_config, train_config, label_version=None):
    datasets = dict()
    for mode in [TRAIN, TEST]:
        feats = list()

        if train_config.feat_keys['nn'] is not None:
            for key in train_config.feat_keys['nn']:
                path = data_config.output_path(key, mode, HIDDEN_FEAT)
                feat_list, _ = load_feat(path)
                feats.append(feat_list)

        if train_config.feat_keys['offline'] is not None:
            for key in train_config.feat_keys['offline']:
                path = data_config.path(mode, FEAT, key)
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
def main(dataset_key, label_version=None, config_path='config_svm.yaml', kernel='rbf'):
    """
    python algo/svm.py main semeval2018_task3 A

    :param dataset_key: string
    :param label_version: string or None
    :param config_path: string
    :param kernel: string
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

    clf = svm.SVC(class_weight=class_weight, kernel=kernel)
    #clf = LogisticRegression(C=1., random_state=0, class_weight='balanced')

    X = datasets[TRAIN][FEATS]
    clf.fit(X=X, y=datasets[TRAIN][LABEL_GOLD])

    if kernel == 'linear':
        coef = sorted(
            list(enumerate(clf.coef_.ravel())),
            key=lambda _item: math.fabs(_item[1])
        )
        coef = map(lambda _item: _item[0], coef)
        print(coef)

    for mode in [TRAIN, TEST]:
        X = datasets[mode][FEATS]
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
    from sklearn import preprocessing
    from sklearn.preprocessing import Normalizer
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

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
        TF: CountVectorizer(
            #tokenizer=lambda x: filter(lambda _t: not _t.startswith('</'), x.split(' ')),
            tokenizer=lambda x: x.split(' '),
            ngram_range=(1, 3),
            min_df=0.02,
            max_features=1000
        ),
        TF_C: TfidfVectorizer(
            ngram_range=(1, 1),
            analyzer='char',
            lowercase=False,
            smooth_idf=True,
            sublinear_tf=True
        )
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
    #clf = LogisticRegression(C=1., random_state=0, class_weight='balanced')

    X = hstack([datasets[TRAIN][k] for k in vectorizers.keys()])
    #scaler = preprocessing.StandardScaler()
    #X = scaler.fit_transform(X=X.todense())
    clf.fit(X=X, y=datasets[TRAIN][LABEL])

    for mode in [TRAIN, TEST]:
        X = hstack([datasets[mode][k] for k in vectorizers.keys()])
        #X = scaler.transform(X=X.todense())
        labels_predict = clf.predict(X=X)
        labels_gold = datasets[mode][LABEL]
        res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)

        print(mode)
        print_evaluation(res)
        print()


@commandr.command('lr')
def logistic_regression(dataset_key, text_version, label_version=None, use_class_weights=True):
    """
    python algo/svm.py lr semeval2018_task3 -t ek -l A
    """
    from sklearn.preprocessing import Normalizer
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

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

    max_features = 10000
    '''vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        #tokenizer=lambda x: x.split(' '),
        #tokenizer=lambda x: x.split(' '),
        analyzer='word',
        min_df=5,
        # max_df=0.9,
        lowercase=False,
        use_idf=True,
        smooth_idf=True,
        max_features=max_features,
        sublinear_tf=True
    )'''

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 6),
        analyzer='char',
        lowercase=False,
        smooth_idf=True,
        #sublinear_tf=True,
        max_features=50000
    )

    clf = LogisticRegression(C=1., random_state=0, class_weight='balanced')
    #clf = svm.SVC(C=0.6, random_state=0, kernel='linear', class_weight='balanced')
    #clf = svm.SVC(C=0.6, random_state=0, kernel='rbf', class_weight='balanced')

    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        #('normalizer', Normalizer(norm='l2')),
        ('clf', clf)
    ])

    pipeline.fit(datasets[TRAIN][TEXT], datasets[TRAIN][LABEL])

    for mode in [TRAIN, TEST]:
        labels_predict = pipeline.predict(datasets[mode][TEXT])
        #print(labels_predict)
        labels_gold = datasets[mode][LABEL]
        res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)

        print(mode)
        print_evaluation(res)
        print()


if __name__ == '__main__':
    commandr.Run()
