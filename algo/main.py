# -*- coding: utf-8 -*-
from __future__ import print_function
import json
import time
import commandr
import numpy as np
import tensorflow as tf
import yaml
import importlib

from algo.model.const import *
from algo.model.train_config import TrainConfig
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
    datasets = dict()
    for mode in [TRAIN, TEST]:
        tokenized_list = load_tokenized_list(TRAIN)
        label_list = load_label_list(TRAIN)

        tid_list = tokenized_to_tid_list(tokenized_list, vocab_id_mapping)
        seq_len_list = seq_to_len_list(tid_list)
        datasets[mode] = {
            TOKEN_ID_SEQ: tid_list,
            SEQ_LEN: np.asarray(seq_len_list),
            LABEL_GOLD: np.asarray(label_list),
        }
        _max_seq_len = max(seq_len_list)
        max_seq_len = _max_seq_len if _max_seq_len > max_seq_len else max_seq_len

    for mode in [TRAIN, TEST]:
        datasets[mode][TOKEN_ID_SEQ] = np.asarray(zero_pad_seq_list(datasets[mode][TOKEN_ID_SEQ], max_seq_len + 1))

    output_dim = max(datasets[TRAIN][LABEL_GOLD]) + 1
    return datasets, output_dim, max_seq_len


dataset_key = {
    TRAIN: [TOKEN_ID_SEQ, SEQ_LEN, LABEL_GOLD],
    TEST: [TOKEN_ID_SEQ, SEQ_LEN]
}

fetch_key = {
    TRAIN: [OPTIMIZER, LOSS, LABEL_PREDICT],
    TEST: [LABEL_PREDICT, HIDDEN_FEAT]
}


@commandr.command
def main():
    config_data = yaml.load(open('config.yaml'))
    module_relative_path = config_data['module']
    NNModel = getattr(importlib.import_module(module_relative_path), 'NNModel')
    NNConfig = getattr(importlib.import_module(module_relative_path), 'NNConfig')

    output_key = '{}_{}'.format(NNModel.name, int(time.time()))
    print('OUTPUT_KEY: {}'.format(output_key))

    # 加载词向量与相关数据
    lookup_table, vocab_id_mapping, embedding_dim = load_lookup_table()
    # 加载训练数据
    datasets, output_dim, max_seq_len = load_dataset(vocab_id_mapping=vocab_id_mapping)
    # 加载配置
    nn_config = NNConfig(config_data['nn'])
    train_config = TrainConfig(config_data['train'])

    index_iterators = {mode: IndexIterator(datasets[mode][LABEL_GOLD]) for mode in [TRAIN, TEST]}
    index_iterators[TRAIN].split_train_valid(train_config.valid_rate)

    if train_config.use_class_weights:
        label_weight = {_label: 1. / len(_index)
            for _label, _index in index_iterators[TRAIN].label_index.items()}
    else:
        label_weight = {_label: 1. for _label in index_iterators[TRAIN].dim}

    # 基于加载的数据更新配置
    nn_config.set_embedding_dim(embedding_dim)
    nn_config.set_output_dim(output_dim)
    nn_config.set_seq_len(max_seq_len + 1)

    # 搭建神经网络
    nn = NNModel(config=nn_config)
    nn.build_neural_network(lookup_table=lookup_table)

    batch_size = train_config.batch_size
    fetches = {mode: {_key: nn.var(_key) for _key in fetch_key[mode]} for mode in [TRAIN, TEST]}

    last_eval = {TRAIN: None, VALID: None, TEST: None}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        dataset = datasets[TRAIN]
        index_iterator = index_iterators[TRAIN]

        for epoch in range(train_config.epoch):
            print('== epoch {} =='.format(epoch))

            # TRAIN SET
            print('TRAIN')
            n_sample = index_iterator.n_sample(TRAIN)
            labels_predict = list()
            labels_gold = list()

            for batch_index in index_iterator.iterate(batch_size, mode=TRAIN, shuffle=True):
                feed_dict = {nn.var(_key): dataset[_key][batch_index] for _key in dataset_key[TRAIN]}
                feed_dict[nn.var(DROPOUT_KEEP_PROB)] = train_config.dropout_keep_prob
                feed_dict[nn.var(SAMPLE_WEIGHTS)] = map(label_weight.get, feed_dict[nn.var(LABEL_GOLD)])
                res = sess.run(fetches=fetches[TRAIN], feed_dict=feed_dict)

                labels_predict += res[LABEL_PREDICT].tolist()
                labels_gold += dataset[LABEL_GOLD][batch_index].tolist()

            labels_predict, labels_gold = labels_predict[:n_sample], labels_gold[:n_sample]
            labels_predict, labels_gold = labels_predict[:n_sample], labels_gold[:n_sample]
            res = basic_evaluate(gold=labels_gold, pred=labels_predict)
            for _key in [F1_SCORE, ACCURACY, PRECISION, RECALL]:
                print('[{}] {}'.format(_key, res[_key]))
            last_eval[TRAIN] = res

            # VALIDATION SET
            print('VALID')
            n_sample = index_iterator.n_sample(VALID)
            labels_predict = list()
            labels_gold = list()

            for batch_index in index_iterator.iterate(batch_size, mode=VALID, shuffle=False):
                feed_dict = {nn.var(_key): dataset[_key][batch_index] for _key in dataset_key[TEST]}
                feed_dict[nn.var(DROPOUT_KEEP_PROB)] = 1.
                res = sess.run(fetches=fetches[TEST], feed_dict=feed_dict)
                labels_predict += res[LABEL_PREDICT].tolist()
                labels_gold += dataset[LABEL_GOLD][batch_index].tolist()

            labels_predict, labels_gold = labels_predict[:n_sample], labels_gold[:n_sample]
            res = basic_evaluate(gold=labels_gold, pred=labels_predict)
            for _key in [F1_SCORE, ACCURACY, PRECISION, RECALL]:
                print('[{}] {}'.format(_key, res[_key]))
            last_eval[VALID] = res

        data_config.prepare_output_folder()

        for mode in [TRAIN, TEST]:
            dataset = datasets[mode]
            index_iterator = index_iterators[mode]
            n_sample = index_iterator.n_sample()

            labels_predict = list()
            labels_gold = list()
            hidden_feats = list()
            for batch_index in index_iterator.iterate(batch_size, shuffle=False):
                feed_dict = {nn.var(_key): dataset[_key][batch_index] for _key in dataset_key[TEST]}
                feed_dict[nn.var(DROPOUT_KEEP_PROB)] = 1.
                res = sess.run(fetches=fetches[TEST], feed_dict=feed_dict)
                labels_predict += res[LABEL_PREDICT].tolist()
                hidden_feats += res[HIDDEN_FEAT].tolist()
                labels_gold += dataset[LABEL_GOLD][batch_index].tolist()

            labels_predict, labels_gold = labels_predict[:n_sample], labels_gold[:n_sample]
            res = basic_evaluate(gold=labels_gold, pred=labels_predict)
            for _key in [F1_SCORE, ACCURACY, PRECISION, RECALL]:
                print('[{}] {}'.format(_key, res[_key]))

            if mode == TEST:
                last_eval[TEST] = res

            with open(data_config.output_path(output_key, mode, HIDDEN_FEAT), 'w') as file_obj:
                for feat in hidden_feats:
                    file_obj.write('\t'.join(map(str, feat)) + '\n')

        for mode in [TRAIN, VALID, TEST]:
            json.dump(last_eval[mode], open(data_config.output_path(output_key, mode, EVALUATION), 'w'))


if __name__ == '__main__':
    commandr.Run()
