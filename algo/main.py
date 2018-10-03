# -*- coding: utf-8 -*-
from __future__ import print_function
import json

import commandr
import numpy as np
import tensorflow as tf
import yaml

from algo.model.const import *
from algo.model.train_config import TrainConfig
from algo.nn.gru import NNModel, NNConfig
from algo.lib.dataset import IndexIterator
from algo.lib.evaluate import basic_evaluate
from dataset.common.const import *
from dataset.semeval2018.task3.config import config as data_config
from nlp.process import naive_tokenize
from nlp.word2vec import PlainModel


def load_text_list(mode):
    path = data_config.path(mode, TEXT)
    text_list = list()
    with open(path, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == '':
                continue
            text = line
            text_list.append(text)
    return text_list


def load_label_list(mode):
    path = data_config.path(mode, LABEL)
    label_list = list()
    with open(path, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == '':
                continue
            label = int(line)
            label_list.append(label)
    return label_list


def load_tokenized_list(mode):
    return map(naive_tokenize, load_text_list(mode))


def load_vocab_list(mode):
    vocab_list = list()
    path = data_config.path(mode, VOCAB, 'v0')
    with open(path, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == '':
                continue
            data = json.loads(line)
            vocab = data['t']
            vocab_list.append(vocab)
    return vocab_list


def normalize_lookup_table(table):
    return (table - table.mean(axis=0)) / table.std(axis=0)


def tokenized_to_tid_list(tokenized_list, vocab_id_mapping):
    tid_list = map(
        lambda _tokens: filter(
            lambda _tid: _tid is not None,
            map(lambda _t: vocab_id_mapping.get(_t), _tokens)
        ),
        tokenized_list
    )
    return tid_list


def load_lookup_table():
    w2v_model = PlainModel(data_config.path(ALL, WORD2VEC, 'google_v0'))
    vocab_list = w2v_model.index.keys()

    train_vocabs = set(load_vocab_list(TRAIN))
    not_supported_vocabs = filter(lambda _vocab: _vocab not in set(vocab_list), train_vocabs)
    n_not_supported = len(not_supported_vocabs)

    lookup_table_pretrained = np.asarray([w2v_model.get(_vocab) for _vocab in vocab_list])
    lookup_table_patch = np.random.random((n_not_supported, w2v_model.dim))

    lookup_table = np.concatenate(
        map(normalize_lookup_table, [lookup_table_pretrained, lookup_table_patch])
    )

    vocab_list += not_supported_vocabs
    vocab_id_mapping = {_vocab: _i for _i, _vocab in enumerate(vocab_list)}

    return lookup_table, vocab_id_mapping, w2v_model.dim


def load_dataset(vocab_id_mapping):
    def seq_to_len_list(seq_list):
        return map(len, seq_list)

    def zero_pad_seq_list(seq_list, seq_len):
        return map(lambda _seq: _seq + [0] * (seq_len - len(_seq)), seq_list)

    max_seq_len = -1
    dataset = dict()
    for mode in [TRAIN, TEST]:
        tokenized_list = load_tokenized_list(TRAIN)
        label_list = load_label_list(TRAIN)

        tid_list = tokenized_to_tid_list(tokenized_list, vocab_id_mapping)
        seq_len_list = seq_to_len_list(tid_list)
        dataset[mode] = {
            TOKEN_ID_SEQ: tid_list,
            SEQ_LEN: np.asarray(seq_len_list),
            LABEL_GOLD: np.asarray(label_list),
        }
        _max_seq_len = max(seq_len_list)
        max_seq_len = _max_seq_len if _max_seq_len > max_seq_len else max_seq_len

    for mode in [TRAIN, TEST]:
        dataset[mode][TOKEN_ID_SEQ] = np.asarray(zero_pad_seq_list(dataset[mode][TOKEN_ID_SEQ], max_seq_len + 1))

    output_dim = max(dataset[TRAIN][LABEL_GOLD]) + 1
    return dataset, output_dim, max_seq_len


@commandr.command
def train():
    # 加载词向量与相关数据
    lookup_table, vocab_id_mapping, embedding_dim = load_lookup_table()
    # 加载训练数据
    dataset, output_dim, max_seq_len = load_dataset(vocab_id_mapping=vocab_id_mapping)
    # 加载配置
    config_data = yaml.load(open('config.yaml'))
    nn_config = NNConfig(config_data['nn'])
    train_config = TrainConfig(config_data['train'])

    train_iterator = IndexIterator(dataset[TRAIN][LABEL_GOLD])
    train_iterator.split_train_valid(train_config.valid_rate)
    label_count = train_iterator.label_count()

    test_iterator = IndexIterator(dataset[TEST][LABEL_GOLD])

    # 基于加载的数据更新配置
    nn_config.set_embedding_dim(embedding_dim)
    nn_config.set_output_dim(output_dim)
    nn_config.set_seq_len(max_seq_len + 1)

    # 搭建神经网络
    nn = NNModel(config=nn_config)
    nn.build_neural_network(lookup_table=lookup_table)

    train_dataset_key = [TOKEN_ID_SEQ, SEQ_LEN, LABEL_GOLD]
    train_fetches = {_key: nn.var(_key) for _key in [OPTIMIZER, LOSS, LABEL_PREDICT, PROB_PREDICT]}

    test_dataset_key = [TOKEN_ID_SEQ, SEQ_LEN]
    test_fetches = {_key: nn.var(_key) for _key in [LABEL_PREDICT, PROB_PREDICT]}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(train_config.epoch):
            print('== epoch {} =='.format(epoch))
            print('TRAIN')
            # TRAIN SET
            labels_predict = list()
            labels_gold = list()
            for batch_index in train_iterator.iterate(train_config.batch_size, mode=TRAIN, shuffle=True):
                feed_dict = {nn.var(_key): dataset[TRAIN][_key][batch_index] for _key in train_dataset_key}
                feed_dict[nn.var(DROPOUT_KEEP_PROB)] = train_config.dropout_keep_prob

                if train_config.use_class_weights:
                    feed_dict[nn.var(SAMPLE_WEIGHTS)] = map(
                        lambda _label: 1. / label_count.get(_label), feed_dict[nn.var(LABEL_GOLD)])
                else:
                    feed_dict[nn.var(SAMPLE_WEIGHTS)] = [1.] * train_config.batch_size

                res = sess.run(fetches=train_fetches, feed_dict=feed_dict)
                labels_predict += res[LABEL_PREDICT].tolist()
                labels_gold += dataset[TRAIN][LABEL_GOLD][batch_index].tolist()

            labels_predict = labels_predict[:train_iterator.n_sample(TRAIN)]
            labels_gold = labels_gold[:train_iterator.n_sample(TRAIN)]
            res = basic_evaluate(gold=labels_gold, pred=labels_predict)

            for _key in [F1_SCORE, ACCURACY, PRECISION, RECALL]:
                print('[{}] {}'.format(_key, res[_key]))

            # VALIDATION SET
            print('VALID')
            labels_predict = list()
            labels_gold = list()
            for batch_index in train_iterator.iterate(train_config.batch_size, mode=VALID, shuffle=True):
                feed_dict = {nn.var(_key): dataset[TRAIN][_key][batch_index] for _key in test_dataset_key}
                feed_dict[nn.var(DROPOUT_KEEP_PROB)] = 1.
                res = sess.run(fetches=test_fetches, feed_dict=feed_dict)
                labels_predict += res[LABEL_PREDICT].tolist()
                labels_gold += dataset[TRAIN][LABEL_GOLD][batch_index].tolist()

            labels_predict = labels_predict[:train_iterator.n_sample(VALID)]
            labels_gold = labels_gold[:train_iterator.n_sample(VALID)]
            res = basic_evaluate(gold=labels_gold, pred=labels_predict)

            for _key in [F1_SCORE, ACCURACY, PRECISION, RECALL]:
                print('[{}] {}'.format(_key, res[_key]))

        print('TEST')
        labels_predict = list()
        labels_gold = list()
        for batch_index in test_iterator.iterate(train_config.batch_size, shuffle=False):
            feed_dict = {nn.var(_key): dataset[TEST][_key][batch_index] for _key in test_dataset_key}
            feed_dict[nn.var(DROPOUT_KEEP_PROB)] = 1.
            res = sess.run(fetches=test_fetches, feed_dict=feed_dict)
            labels_predict += res[LABEL_PREDICT].tolist()
            labels_gold += dataset[TEST][LABEL_GOLD][batch_index].tolist()

        labels_predict = labels_predict[:test_iterator.n_sample()]
        labels_gold = labels_gold[:test_iterator.n_sample()]
        res = basic_evaluate(gold=labels_gold, pred=labels_predict)
        for _key in [F1_SCORE, ACCURACY, PRECISION, RECALL]:
            print('[{}] {}'.format(_key, res[_key]))


if __name__ == '__main__':
    commandr.Run()
