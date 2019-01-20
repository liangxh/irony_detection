# -*- coding: utf-8 -*-
import numpy as np
from algo.model.const import *
from nlp.word2vec import PlainModel

keys = [ACCURACY, PRECISION, RECALL, F1_SCORE, 'test_score']


def print_evaluation(res):
    values = list(map(res.get, keys))
    print(','.join(keys))
    print(','.join(list(map(str, values))))


def print_evaluation_0(res):
    keys_ = [RECALL_0, PRECISION_0]
    values = list(map(res.get, keys_))
    print(','.join(keys))
    print(','.join(list(map(str, values))))


def tid_dropout(tids, dropout_keep_rate):
    tids = np.asarray(tids)
    mask = (np.random.random(tids.shape) <= dropout_keep_rate).astype(int)
    return tids * mask


def tokenized_to_tid_list(tokenized_list, vocab_id_mapping):
    tid_list = list(map(
        lambda _tokens: list(filter(
            lambda _tid: _tid is not None,
            list(map(lambda _t: vocab_id_mapping.get(_t), _tokens))
        )),
        tokenized_list
    ))
    return tid_list


def load_lookup_table(w2v_model_path, vocabs):
    """
    取模型的所有词向量, 以及为train中出现但模型中没有的词随机生成词向量
    :param w2v_model_path: string
    :param vocabs: list of string, 需要确保在的字典, 不在模型中的话则随机生成
    :return:
        lookup_table: np.array, [VOCAB_SIZE, EMBEDDING_DIM]
        vocab_id_mapping: dict, <VOCAB, ID>
        embedding_dim: int
    """
    w2v_model = PlainModel(w2v_model_path)
    vocab_list = list(w2v_model.index.keys())

    not_supported_vocabs = list(filter(lambda _vocab: _vocab not in set(vocab_list), vocabs))
    n_not_supported = len(not_supported_vocabs)

    lookup_table_pretrained = np.asarray([w2v_model.get(_vocab) for _vocab in vocab_list])
    print('n supported: {}'.format(lookup_table_pretrained.shape[0]))
    print('n not supported: {}'.format(n_not_supported))

    table_mean = lookup_table_pretrained.mean(axis=0)
    table_std = (lookup_table_pretrained - table_mean).std(axis=0)

    lookup_table_patch = np.random.normal(
        table_mean, table_std, (n_not_supported, w2v_model.dim))

    lookup_table = np.concatenate([lookup_table_pretrained, lookup_table_patch])

    vocab_list += not_supported_vocabs
    vocab_id_mapping = {_vocab: _i for _i, _vocab in enumerate(vocab_list)}

    embedding_dim = w2v_model.dim
    return lookup_table, vocab_id_mapping, embedding_dim


def load_lookup_table2(w2v_model_path, vocabs):
    """
    取模型的所有词向量, 以及为train中出现但模型中没有的词随机生成词向量
    :param w2v_model_path: string
    :param vocabs: list of string, 需要确保在的字典, 不在模型中的话则随机生成
    :return:
        lookup_table: np.array, [VOCAB_SIZE, EMBEDDING_DIM]
        vocab_id_mapping: dict, <VOCAB, ID>
        embedding_dim: int
    """
    w2v_model = PlainModel(w2v_model_path)
    vocab_list = list(w2v_model.index.keys())

    not_supported_vocabs = list(filter(lambda _vocab: _vocab not in set(vocab_list), vocabs))
    n_not_supported = len(not_supported_vocabs)

    lookup_table_pretrained = np.asarray([w2v_model.get(_vocab) for _vocab in vocab_list])
    print('n supported: {}'.format(lookup_table_pretrained.shape[0]))
    print('n not supported: {}'.format(n_not_supported))

    table_mean = lookup_table_pretrained.mean(axis=0)
    table_std = (lookup_table_pretrained - table_mean).std(axis=0)
    zero_vec = np.zeros((1, w2v_model.dim))

    lookup_table_patch = np.random.normal(
        table_mean, table_std, (n_not_supported, w2v_model.dim))

    lookup_table = np.concatenate([zero_vec, lookup_table_pretrained, lookup_table_patch])

    vocab_list = ['ZERO_VEC', ] + vocab_list + not_supported_vocabs
    vocab_id_mapping = {_vocab: _i for _i, _vocab in enumerate(vocab_list)}

    embedding_dim = w2v_model.dim
    return lookup_table, vocab_id_mapping, embedding_dim


def build_random_lookup_table(vocabs, dim):
    vocab_list = list(vocabs)
    lookup_table = np.random.normal(0, 1, (len(vocabs), dim))
    vocab_id_mapping = {_vocab: _i for _i, _vocab in enumerate(vocab_list)}
    return lookup_table, vocab_id_mapping, dim


def generate_wrong_prediction_report(labels_gold, labels_predict, text_list):
    res = list()
    for l_gold, l_predict, t in zip(labels_gold, labels_predict, text_list):
        if l_predict != l_gold:
            res.append((l_gold, l_predict, t))
    res = sorted(res, key=lambda _item: (_item[0], _item[1]))
    return res
