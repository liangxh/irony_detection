# -*- coding: utf-8 -*-
import numpy as np
from algo.model.const import *
from nlp.word2vec import PlainModel

keys = [ACCURACY, PRECISION, RECALL, F1_SCORE]


def print_evaluation(res):
    values = map(res.get, keys)
    print('\t'.join(map(lambda _k: _k + ' ' * (10 - len(_k)), keys)))
    print('\t'.join(map(str, values)))


def tokenized_to_tid_list(tokenized_list, vocab_id_mapping):
    tid_list = map(
        lambda _tokens: filter(
            lambda _tid: _tid is not None,
            map(lambda _t: vocab_id_mapping.get(_t), _tokens)
        ),
        tokenized_list
    )
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
    vocab_list = w2v_model.index.keys()

    not_supported_vocabs = filter(lambda _vocab: _vocab not in set(vocab_list), vocabs)
    n_not_supported = len(not_supported_vocabs)

    lookup_table_pretrained = np.asarray([w2v_model.get(_vocab) for _vocab in vocab_list])
    lookup_table_patch = np.random.random((n_not_supported, w2v_model.dim))

    def normalize_lookup_table(table):
        return (table - table.mean(axis=0)) / table.std(axis=0)

    lookup_table = np.concatenate(
        map(normalize_lookup_table, [lookup_table_pretrained, lookup_table_patch])
    )

    vocab_list += not_supported_vocabs
    vocab_id_mapping = {_vocab: _i for _i, _vocab in enumerate(vocab_list)}

    embedding_dim = w2v_model.dim
    return lookup_table, vocab_id_mapping, embedding_dim