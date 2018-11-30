# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import importlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from dataset.common.const import *
from dataset.common.load import *
from nlp.word2vec import PlainModel

NBOW = 'nbow'
SUM = 'sum'
MEAN = 'mean'
MIN = 'min'
MAX = 'max'


@commandr.command('tf')
def calculate_tf(dataset_key, text_version):
    """
    [Usage]
    python tools/n_gram.py tf -d semeval2018_task3 -t ek

    :param dataset_key: string
    :param text_version: string
    :return:
    """
    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')
    texts = dict()
    for mode in [TRAIN, TEST]:
        texts[mode] = load_text_list(data_config.path(mode, TEXT, text_version))

    output_version = '{}_{}'.format('tf', text_version)
    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=0.02)

    feat = vectorizer.fit_transform(texts[TRAIN])
    path = data_config.path(TRAIN, FEAT, output_version)
    with open(path, 'w') as file_obj:
        for vec in feat.todense().tolist():
            file_obj.write('\t'.join(map(str, vec)) + '\n')

    feat = vectorizer.transform(texts[TEST])
    path = data_config.path(TEST, FEAT, output_version)
    with open(path, 'w') as file_obj:
        for vec in feat.todense().tolist():
            file_obj.write('\t'.join(map(str, vec)) + '\n')


@commandr.command('nbow')
def calculate_nbow(dataset_key, text_version, w2v_version, nbow_mode):
    """
    [Usage]
    python nlp/bow.py nbow -d semeval2018_task3 -t ek -w google -n min
    python nlp/bow.py nbow -d semeval2018_task3 -t ek -w ntua -n mean

    :param dataset_key: string
    :param text_version: string
    :param w2v_version: string
    :param nbow_mode: string
    :return:
    """
    w2v_key = '{}_{}'.format(w2v_version, text_version)

    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')
    w2v_model_path = data_config.path(ALL, WORD2VEC, w2v_key)
    w2v_model = PlainModel(w2v_model_path)

    for mode in [TRAIN, TEST]:
        text_path = data_config.path(mode, TEXT, text_version)
        tokenized_list = load_tokenized_list(text_path)

        output_key = '{}_{}_{}'.format(NBOW, w2v_key, nbow_mode)
        path_feat = data_config.path(mode, FEAT, output_key)

        with open(path_feat, 'w') as fobj:
            for tokens in tokenized_list:
                vecs = list(map(w2v_model.get, tokens))
                vecs = filter(lambda _v: _v is not None, vecs)

                if len(vecs) == 0:
                    feat = np.asarray([0.] * w2v_model.dim)
                    print(' '.join(tokens))
                else:
                    vecs = np.asarray(vecs)

                    if nbow_mode == MIN:
                        feat = np.amin(vecs, axis=0)
                    elif nbow_mode == MAX:
                        feat = np.amax(vecs, axis=0)
                    elif nbow_mode == SUM:
                        feat = np.sum(vecs, axis=0)
                    elif nbow_mode == MEAN:
                        feat = np.mean(vecs, axis=0)
                    else:
                        raise ValueError('invalid nbow_mode: {}'.format(nbow_mode))

                fobj.write('\t'.join(map(str, feat.tolist())) + '\n')


if __name__ == '__main__':
    commandr.Run()
