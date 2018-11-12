# -*- coding: utf-8 -*-
import commandr
import importlib
import numpy as np
from dataset.common.const import *
from dataset.common.load import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer


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

    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=0.02)

    feat = vectorizer.fit_transform(texts[TRAIN])
    path = data_config.path(TRAIN, FEAT, 'tf')
    with open(path, 'w') as file_obj:
        for vec in feat.todense().tolist():
            file_obj.write('\t'.join(map(str, vec)) + '\n')

    feat = vectorizer.transform(texts[TEST])
    path = data_config.path(TEST, FEAT, 'tf')
    with open(path, 'w') as file_obj:
        for vec in feat.todense().tolist():
            file_obj.write('\t'.join(map(str, vec)) + '\n')


if __name__ == '__main__':
    commandr.Run()
