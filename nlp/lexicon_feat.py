# -*- coding: utf-8 -*-
import commandr
import importlib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from dataset.common.const import *
from dataset.common.load import *


def extract_tf(dataset_key, text_version):
    #vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer(max_df=0.5)
    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')

    texts = dict()
    for mode in [TRAIN, TEST]:
        texts[mode] = load_text_list(data_config.path(mode, TEXT, text_version))

    feat = dict()
    feat[TRAIN] = vectorizer.fit_transform(texts[TRAIN])
    feat[TEST] = vectorizer.transform(texts[TEST])
    return feat


@commandr.command
def tf_idf(dataset_key, text_version):
    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')
    for mode in [TRAIN, TEST]:
        text_path = data_config.path(mode, text_version)
        tokenized_list = load_tokenized_list(text_path)

    datasets = dict()
    for mode in [TRAIN, TEST]:
        datasets[mode] = {
            TEXT: load_text_list(data_config.path(mode, TEXT, text_version)),
            LABEL: load_label_list(data_config.path(mode, LABEL, label_version))
        }

    vectorizers = {
        #TF_IDF: TfidfVectorizer(),
        TF: CountVectorizer()
    }

    for key, vectorizer in vectorizers.items():
        datasets[TRAIN][key] = vectorizer.fit_transform(datasets[TRAIN][TEXT])
        datasets[TEST][key] = vectorizer.transform(datasets[TEST][TEXT])


