# -*- coding: utf-8 -*-
import commandr
import importlib
import math
import numpy as np
import string
from dataset.common.const import *
from nlp.word2vec import PlainModel
from dataset.common.load import load_tokenized_list

MAX = 'max'
MIN = 'min'

DOT = 'dot'
COS = 'cos'
DIST = 'dist'


class SentenceEmbeddingSimilarity(object):
    def __init__(self, words, vecs, mode):
        assert len(words) == len(vecs)
        self.mode = mode
        self.words = words
        self.vecs = map(lambda vec: np.asarray(vec) if vec is not None else None, vecs)
        self.length = len(self.words)
        self.sim_cache = dict()

    def calculate_similarity(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return None
        else:
            if self.mode == DOT:
                return np.dot(vec1, vec2)

            if self.mode == COS:
                vec1 /= np.sqrt(vec1 ** 2).sum(axis=0)
                vec2 /= np.sqrt(vec2 ** 2).sum(axis=0)
                return np.dot(vec1, vec2)

            raise ValueError('unknown mode: {}'.format(self.mode))

    def sim(self, i, j, weighted):
        if i == j or self.words[i] == self.words[j]:
            return None

        if i > j:
            i, j = j, i
        key = '{}_{}'.format(i, j)
        if key not in self.sim_cache:
            self.sim_cache[key] = self.calculate_similarity(self.vecs[i], self.vecs[j])

        if self.sim_cache[key] is None:
            return None

        value = float(self.sim_cache[key])
        if weighted:
            value /= (j - i)  # TODO 可以考虑测试不同权重方案，原方案为间隔的平方
        return value

    def analyse(self, weighted=False):
        similar = {MAX: None, MIN: None}
        dissimilar = {MAX: None, MIN: None}

        def update_record(record, v):
            if v is None:
                raise ValueError('v should not be None')
            if record[MAX] is None or record[MAX] < v:
                record[MAX] = v
            if record[MIN] is None or record[MIN] > v:
                record[MIN] = v

        for i in range(self.length):
            sim_list = [self.sim(i, j, weighted=weighted) for j in range(self.length)]
            sim_list = filter(lambda _v: _v is not None, sim_list)
            if len(sim_list) == 0.:
                continue
            max_value, min_value = max(sim_list), min(sim_list)
            update_record(record=similar, v=max_value)
            update_record(record=dissimilar, v=min_value)

        return similar, dissimilar


@commandr.command('embedding')
def embedding_incongruity(dataset_key, text_version, w2v_version, sim_mode=COS):
    """
    [Usage]
    python nlp/incongruity.py embedding -d semeval2018_task3 -t ek -w google -s dot

    python nlp/incongruity.py embedding -d semeval2018_task3 -t ek -w google
    python nlp/incongruity.py embedding -d semeval2018_task3 -t ek -w glove_25
    python nlp/incongruity.py embedding -d semeval2018_task3 -t ek -w glove_50
    python nlp/incongruity.py embedding -d semeval2018_task3 -t ek -w glove_100
    python nlp/incongruity.py embedding -d semeval2018_task3 -t ek -w glove_200

    :param dataset_key: string
    :param text_version: string
    :param w2v_version: string
    :param sim_mode: string
    :return:
    """
    w2v_key = '{}_{}'.format(w2v_version, text_version)

    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')
    w2v_model_path = data_config.path(ALL, WORD2VEC, w2v_key)
    w2v_model = PlainModel(w2v_model_path)

    punctuations = set(string.punctuation)

    for mode in [TRAIN, TEST]:
        text_path = data_config.path(mode, TEXT, text_version)
        tokenized_list = load_tokenized_list(text_path)

        output_key = '{}.{}'.format(EMBEDDING_INCONGRUITY, w2v_key)
        if sim_mode != COS:
            output_key += '_{}'.format(sim_mode)
        path_feat = data_config.path(mode, FEAT, output_key)

        output_key = '{}.{}'.format(EMBEDDING_INCONGRUITY_WEIGHTED, w2v_key)
        if sim_mode != COS:
            output_key += '_{}'.format(sim_mode)
        path_feat_weighted = data_config.path(mode, FEAT, output_key)

        with open(path_feat, 'w') as fobj, open(path_feat_weighted, 'w') as fobj_weighted:
            for tokens in tokenized_list:
                tokens = filter(lambda _t: _t not in punctuations, tokens)

                vecs = map(w2v_model.get, tokens)
                record = SentenceEmbeddingSimilarity(words=tokens, vecs=vecs, mode=sim_mode)

                similar, dissimilar = record.analyse(weighted=False)
                feat = [similar[MAX], similar[MIN], dissimilar[MAX], dissimilar[MIN]]
                if feat == [None] * 4:
                    feat = [0.] * 4
                fobj.write('\t'.join(map(str, feat)) + '\n')

                similar, dissimilar = record.analyse(weighted=True)
                feat = [similar[MAX], similar[MIN], dissimilar[MAX], dissimilar[MIN]]
                if feat == [None] * 4:
                    feat = [0.] * 4
                fobj_weighted.write('\t'.join(map(str, feat)) + '\n')


if __name__ == '__main__':
    commandr.Run()
