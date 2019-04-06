# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import time
import copy
import commandr
import yaml
import random
import shutil
import importlib
import numpy as np
import tensorflow as tf
from algo.lib.dataset import IndexIterator, SimpleIndexIterator
from algo.lib.evaluate93 import basic_evaluate
from algo.model.const import *
from algo.model.train_config import TrainConfig
from algo.lib.common import print_evaluation, load_lookup_table2, tokenized_to_tid_list
from algo.model.nn_config import BaseNNConfig
from algo.nn.base import BaseNNModel
from dataset.common.const import *
from dataset.common.load import *
from dataset.semeval2019_task3_dev.config import config as data_config


class NNConfig(BaseNNConfig):
    @property
    def rnn_dim(self):
        return self.data['rnn']['dim']

    @property
    def attention_dim(self):
        return self.data['attention']['dim']

    @property
    def filter_num(self):
        return self.data['cnn']['filter_num']

    @property
    def kernel_size(self):
        return self.data['cnn']['kernel_size']


def to_nn_input(tid_list, max_seq_len):
    tid_list = trim_tid_list(tid_list, max_seq_len)
    seq_len = np.asarray(list(map(lambda _seq: min(max_seq_len, len(_seq) + 1), tid_list)))
    tid_list = np.asarray(zero_pad_seq_list(tid_list, max_seq_len))
    return tid_list, seq_len


def load_dataset(mode, vocab_id_mapping, max_seq_len, sampling=False, label_map=None, with_label=True, label_version=None):
    modes = mode if isinstance(mode, list) else [mode, ]

    dataset = dict()
    for i in [0, 1, 2]:
        tid_list = list()
        for mode in modes:
            tid_list += tokenized_to_tid_list(
                load_tokenized_list(data_config.path(mode, TURN, '{}.ek'.format(i))),
                vocab_id_mapping
            )
        dataset[TID_[i]] = tid_list

    if with_label:
        label_list = list()
        for mode in modes:
            label_path = data_config.path(mode, LABEL, label_version)
            label_list += load_label_list(label_path)

        if label_map is not None:
            new_tid_list_ = [list() for _ in range(3)]
            new_label_list = list()
            for idx, label in enumerate(label_list):
                if label in label_map:
                    for i in range(3):
                        new_tid_list_[i].append(dataset[TID_[i]][idx])
                    new_label_list.append(label_map[label])
            for i in range(3):
                dataset[TID_[i]] = new_tid_list_[i]
            label_list = new_label_list

        dataset[LABEL_GOLD] = label_list

    if sampling:
        dataset = custom_sampling(dataset)

    for i in [0, 1, 2]:
        dataset[TID_[i]], dataset[SEQ_LEN_[i]] = to_nn_input(dataset[TID_[i]], max_seq_len=max_seq_len)

    if with_label:
        dataset[LABEL_GOLD] = np.asarray(dataset[LABEL_GOLD])
        output_dim = max(dataset[LABEL_GOLD]) + 1
        return dataset, output_dim
    else:
        return dataset


def custom_sampling(dataset):
    label_idx = [list() for _ in range(4)]
    for i, label in enumerate(dataset[LABEL_GOLD]):
        label_idx[label].append(i)

    label = 0
    for i in label_idx[label]:
        tid_ = [copy.deepcopy(dataset[TID_[j]][i]) for j in range(3)]

        j = random.randint(0, 2)
        if len(tid_[j]) > 1:
            pop_idx = random.randint(0, len(tid_[j]) - 1)
            tid_[j].pop(pop_idx)

        for j in range(3):
            dataset[TID_[j]].append(tid_[j])
        dataset[LABEL_GOLD].append(label)

    return dataset


feed_key = {
    TRAIN: [LABEL_GOLD] + TID_ + SEQ_LEN_,
    TEST: [] + TID_ + SEQ_LEN_,
}

fetch_key = {
    TRAIN: [OPTIMIZER, LOSS, LABEL_PREDICT],
    TEST: [LABEL_PREDICT, PROB_PREDICT]
}


@commandr.command
def train(model_name, label_version=None, label_key=None, config_path='c93f.yaml'):
    """
    python -m algo.main93_v2 train
    python3 -m algo.main93_v2 train -c config_ntua93.yaml

    :param model_name: string
    :param label_version: string
    :param config_path: string
    :return:
    """
    text_version = 'ek'
    config_data = yaml.load(open(config_path))
    NNModel = getattr(importlib.import_module('algo.m93.{}'.format(model_name)), 'NNModel')

    output_key = 'f_{}_{}_{}'.format(model_name, text_version, int(time.time()))
    if label_version is not None:
        output_key = '{}_{}'.format(label_version, output_key)
    print('OUTPUT_KEY: {}'.format(output_key))

    # 准备输出路径的文件夹
    data_config.prepare_output_folder(output_key=output_key)
    data_config.prepare_model_folder(output_key=output_key)

    shutil.copy(config_path, data_config.output_path(output_key, ALL, CONFIG))

    w2v_key = '{}_{}'.format(config_data['word']['w2v_version'], text_version)
    w2v_model_path = data_config.path(ALL, WORD2VEC, w2v_key)
    vocab_train_path = data_config.path(TRAIN, VOCAB, text_version)

    # 加载字典集
    # 在模型中会采用所有模型中支持的词向量, 并为有足够出现次数的单词随机生成词向量
    vocab_meta_list = load_vocab_list(vocab_train_path)
    vocabs = [_meta['t'] for _meta in vocab_meta_list if _meta['tf'] >= config_data['word']['min_tf']]

    # 加载词向量与相关数据
    lookup_table, vocab_id_mapping, embedding_dim = load_lookup_table2(
        w2v_model_path=w2v_model_path, vocabs=vocabs)
    json.dump(vocab_id_mapping, open(data_config.output_path(output_key, ALL, VOCAB_ID_MAPPING), 'w'))

    # 加载配置
    nn_config = NNConfig(config_data)
    train_config = TrainConfig(config_data['train'])
    early_stop_metric = train_config.early_stop_metric

    # 加载训练数据
    datasets = dict()
    datasets[TRAIN], output_dim = load_dataset(
        mode=[TRAIN, TEST], vocab_id_mapping=vocab_id_mapping, max_seq_len=nn_config.seq_len,
        label_version=label_version, sampling=train_config.train_sampling,
        label_map=train_config.label_map(label_key)
    )
    datasets[TEST], _ = load_dataset(
        mode=FINAL, vocab_id_mapping=vocab_id_mapping, max_seq_len=nn_config.seq_len,
        label_version=label_version, sampling=train_config.train_sampling,
        label_map=train_config.label_map(label_key)
    )

    # 初始化数据集的检索
    index_iterators = {
        TRAIN: IndexIterator.from_dataset(datasets[TRAIN]),
        TEST: IndexIterator.from_dataset(datasets[TEST])
    }
    # 按配置将训练数据切割成训练集和验证集
    index_iterators[TRAIN].split_train_valid(train_config.valid_rate)

    # 计算各个类的权重
    if train_config.use_class_weights:
        label_weight = {
            # 参考 sklearn 中 class_weight='balanced'的公式, 实验显示效果显着
            _label: float(index_iterators[TRAIN].n_sample()) / (index_iterators[TRAIN].dim * len(_index))
            for _label, _index in index_iterators[TRAIN].label_index.items()
        }
    else:
        label_weight = {_label: 1. for _label in range(index_iterators[TRAIN].dim)}

    # 基于加载的数据更新配置
    nn_config.set_embedding_dim(embedding_dim)
    nn_config.set_output_dim(output_dim)
    # 搭建神经网络
    nn = NNModel(config=nn_config)
    nn.build_neural_network(lookup_table=lookup_table)

    batch_size = train_config.batch_size
    fetches = {mode: {_key: nn.var(_key) for _key in fetch_key[mode]} for mode in [TRAIN, TEST]}

    model_output_prefix = data_config.model_path(key=output_key) + '/model'

    best_res = {mode: None for mode in [TRAIN, VALID, TEST]}
    no_update_count = {mode: 0 for mode in [TRAIN, VALID]}
    max_no_update_count = 10

    eval_history = {TRAIN: list(), VALID: list(), TEST: list()}
    best_epoch = -1
    best_epoch_test = -1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        dataset = datasets[TRAIN]
        index_iterator = index_iterators[TRAIN]

        # 训练开始 ##########################################################################
        for epoch in range(train_config.epoch):
            print('== epoch {} = {} ='.format(epoch, output_key))

            # 利用训练集进行训练
            print('TRAIN')
            n_sample = index_iterator.n_sample(TRAIN)
            labels_predict = list()
            labels_gold = list()

            for batch_index in index_iterator.iterate(batch_size, mode=TRAIN, shuffle=True):
                feed_dict = {nn.var(_key): dataset[_key][batch_index] for _key in feed_key[TRAIN]}
                feed_dict[nn.var(SAMPLE_WEIGHTS)] = list(map(label_weight.get, feed_dict[nn.var(LABEL_GOLD)]))
                feed_dict[nn.var(TEST_MODE)] = 0
                res = sess.run(fetches=fetches[TRAIN], feed_dict=feed_dict)

                labels_predict += res[LABEL_PREDICT].tolist()
                labels_gold += dataset[LABEL_GOLD][batch_index].tolist()

            labels_predict, labels_gold = labels_predict[:n_sample], labels_gold[:n_sample]
            res = basic_evaluate(gold=labels_gold, pred=labels_predict)
            print_evaluation(res)
            eval_history[TRAIN].append(res)

            global_step = tf.train.global_step(sess, nn.var(GLOBAL_STEP))

            if train_config.valid_rate == 0.:
                if best_epoch <= 10 or (
                        best_res[TRAIN] is None or res[early_stop_metric] > best_res[TRAIN][early_stop_metric]):
                    best_epoch = epoch
                    best_res[TRAIN] = res
                    no_update_count[TRAIN] = 0
                    saver.save(sess, save_path=model_output_prefix, global_step=global_step)
                else:
                    no_update_count[TRAIN] += 1
            else:
                if best_res[TRAIN] is None or res[early_stop_metric] > best_res[TRAIN][early_stop_metric]:
                    best_res[TRAIN] = res
                    no_update_count[TRAIN] = 0
                else:
                    no_update_count[TRAIN] += 1

                # 计算在验证集上的表现, 不更新模型参数
                print('VALID')
                n_sample = index_iterator.n_sample(VALID)
                labels_predict = list()
                labels_gold = list()

                for batch_index in index_iterator.iterate(batch_size, mode=VALID, shuffle=False):
                    feed_dict = {nn.var(_key): dataset[_key][batch_index] for _key in feed_key[TEST]}
                    feed_dict[nn.var(TEST_MODE)] = 1
                    res = sess.run(fetches=fetches[TEST], feed_dict=feed_dict)

                    labels_predict += res[LABEL_PREDICT].tolist()
                    labels_gold += dataset[LABEL_GOLD][batch_index].tolist()

                labels_predict, labels_gold = labels_predict[:n_sample], labels_gold[:n_sample]
                res = basic_evaluate(gold=labels_gold, pred=labels_predict)
                eval_history[VALID].append(res)
                print_evaluation(res)

                # Early Stop
                if best_epoch <= 10 or (
                        best_res[VALID] is None or res[early_stop_metric] > best_res[VALID][early_stop_metric]):
                    best_epoch = epoch
                    saver.save(sess, save_path=model_output_prefix, global_step=global_step)
                    best_res[VALID] = res
                    no_update_count[VALID] = 0
                else:
                    no_update_count[VALID] += 1

            # eval test
            _mode = TEST
            _dataset = datasets[_mode]
            _index_iterator = SimpleIndexIterator.from_dataset(_dataset)
            _n_sample = _index_iterator.n_sample()

            labels_predict = list()
            labels_gold = list()
            for batch_index in _index_iterator.iterate(batch_size, shuffle=False):
                feed_dict = {nn.var(_key): _dataset[_key][batch_index] for _key in feed_key[TEST]}
                feed_dict[nn.var(TEST_MODE)] = 1
                res = sess.run(fetches=fetches[TEST], feed_dict=feed_dict)

                labels_predict += res[LABEL_PREDICT].tolist()
                labels_gold += _dataset[LABEL_GOLD][batch_index].tolist()
            labels_predict, labels_gold = labels_predict[:_n_sample], labels_gold[:_n_sample]
            res = basic_evaluate(gold=labels_gold, pred=labels_predict)
            eval_history[TEST].append(res)
            print('TEST')
            print_evaluation(res)

            if best_res[TEST] is None or res[F1_SCORE] > best_res[TEST][F1_SCORE]:
                best_res[TEST] = res
                best_epoch_test = epoch

            if no_update_count[TRAIN] >= max_no_update_count:
                break

        # 训练结束 ##########################################################################
        # 确保输出文件夹存在

    print('========================= BEST ROUND EVALUATION =========================')

    json.dump(eval_history, open(data_config.output_path(output_key, 'eval', 'json'), 'w'))

    labels_predict_final = None
    labels_gold_final = load_label_list(data_config.path(FINAL, LABEL))

    with tf.Session() as sess:
        prefix_checkpoint = tf.train.latest_checkpoint(data_config.model_path(key=output_key))
        saver = tf.train.import_meta_graph('{}.meta'.format(prefix_checkpoint))
        saver.restore(sess, prefix_checkpoint)

        nn = BaseNNModel(config=None)
        nn.set_graph(tf.get_default_graph())
        for mode in [TRAIN, TEST, FINAL]:
            dataset = load_dataset(
                mode=mode, vocab_id_mapping=vocab_id_mapping,
                max_seq_len=nn_config.seq_len, with_label=False
            )
            index_iterator = SimpleIndexIterator.from_dataset(dataset)
            n_sample = index_iterator.n_sample()

            prob_predict = list()
            labels_predict = list()

            for batch_index in index_iterator.iterate(batch_size, shuffle=False):
                feed_dict = {nn.var(_key): dataset[_key][batch_index] for _key in feed_key[TEST]}
                feed_dict[nn.var(TEST_MODE)] = 1
                res = sess.run(fetches=fetches[TEST], feed_dict=feed_dict)
                prob_predict += res[PROB_PREDICT].tolist()
                labels_predict += res[LABEL_PREDICT].tolist()

            prob_predict = prob_predict[:n_sample]
            labels_predict = labels_predict[:n_sample]

            if mode == FINAL:
                labels_predict_final = labels_predict

            # 导出预测的label
            with open(data_config.output_path(output_key, mode, LABEL_PREDICT), 'w') as file_obj:
                for _label in labels_predict:
                    file_obj.write('{}\n'.format(_label))
            with open(data_config.output_path(output_key, mode, PROB_PREDICT), 'w') as file_obj:
                for _prob in prob_predict:
                    file_obj.write('\t'.join(map(str, _prob)) + '\n')

    print('====== best epoch test: {} ======'.format(best_epoch_test))

    for mode in [TRAIN, VALID, TEST]:
        if mode == VALID and train_config.valid_rate == 0.:
            continue

        print(mode)
        res = eval_history[mode][best_epoch_test]
        print_evaluation(res)
        for col in res[CONFUSION_MATRIX]:
            print(','.join(map(str, col)))

    print(eval_history[TEST][best_epoch_test])
    print()

    print('====== best epoch valid: {} ======'.format(best_epoch))
    for mode in [TRAIN, VALID, TEST]:
        if mode == VALID and train_config.valid_rate == 0.:
            continue

        print(mode)
        res = eval_history[mode][best_epoch]
        print_evaluation(res)
        for col in res[CONFUSION_MATRIX]:
            print(','.join(map(str, col)))

        json.dump(res, open(data_config.output_path(output_key, mode, EVALUATION), 'w'))
        print()

    print(eval_history[TEST][best_epoch])
    print()

    print('====== label_map check ======')

    label_map = train_config.label_map(label_key)
    if label_map is not None:
        new_gold = list()
        new_pred = list()
        for g, p in zip(labels_gold_final, labels_predict_final):
            if g in label_map:
                new_gold.append(label_map[g])
                new_pred.append(p)
        labels_gold_final = new_gold
        labels_predict_final = new_pred

    res = basic_evaluate(gold=labels_gold_final, pred=labels_predict_final)
    print_evaluation(res)
    for col in res[CONFUSION_MATRIX]:
        print(','.join(map(str, col)))

    print('OUTPUT_KEY: {}'.format(output_key))


@commandr.command('pred')
def predict(output_key, mode):
    config_path = data_config.output_path(output_key, ALL, CONFIG)
    config_data = yaml.load(open(config_path))
    nn_config = NNConfig(config_data)
    vocab_id_mapping = json.load(open(data_config.output_path(output_key, ALL, VOCAB_ID_MAPPING), 'r'))

    dataset = load_dataset(
        mode=mode, vocab_id_mapping=vocab_id_mapping,
        max_seq_len=nn_config.seq_len, sampling=False, with_label=False
    )
    index_iterator = SimpleIndexIterator.from_dataset(dataset)
    n_sample = index_iterator.n_sample()

    with tf.Session() as sess:
        prefix_checkpoint = tf.train.latest_checkpoint(data_config.model_path(key=output_key))
        saver = tf.train.import_meta_graph('{}.meta'.format(prefix_checkpoint))
        saver.restore(sess, prefix_checkpoint)

        nn = BaseNNModel(config=None)
        nn.set_graph(tf.get_default_graph())

        fetches = {_key: nn.var(_key) for _key in [LABEL_PREDICT]}
        labels_predict = list()

        for batch_index in index_iterator.iterate(nn_config.batch_size, shuffle=False):
            feed_dict = {nn.var(_key): dataset[_key][batch_index] for _key in feed_key[TEST]}
            feed_dict[nn.var(TEST_MODE)] = 1
            res = sess.run(fetches=fetches, feed_dict=feed_dict)
            labels_predict += res[LABEL_PREDICT].tolist()

        labels_predict = labels_predict[:n_sample]

    # 导出预测的label
    with open(data_config.output_path(output_key, mode, LABEL_PREDICT), 'w') as file_obj:
        for _label in labels_predict:
            file_obj.write('{}\n'.format(_label))


@commandr.command('eval')
def show_eval(output_key):
    labels_predict_ = dict()
    labels_gold_ = dict()
    for mode in [TRAIN, TEST, FINAL]:
        path = data_config.output_path(output_key, mode, LABEL_PREDICT)
        labels_predict_[mode] = load_label_list(path)

        path = data_config.path(mode, LABEL)
        labels_gold_[mode] = load_label_list(path)

    print('TRAIN + TEST')
    res = basic_evaluate(
        gold=labels_gold_[TRAIN] + labels_gold_[TEST],
        pred=labels_predict_[TRAIN] + labels_predict_[TEST]
    )
    print_evaluation(res)
    for col in res[CONFUSION_MATRIX]:
        print(','.join(map(str, col)))
    print()

    print('FINAL')
    res = basic_evaluate(
        gold=labels_gold_[FINAL],
        pred=labels_predict_[FINAL]
    )
    print_evaluation(res)
    for col in res[CONFUSION_MATRIX]:
        print(','.join(map(str, col)))
    print()


@commandr.command('clear')
def clear_output(output_key):
    shutil.rmtree(data_config.output_folder(output_key))
    shutil.rmtree(data_config.model_folder(output_key))


@commandr.command('conf')
def show_config(output_key):
    path = data_config.output_path(output_key, ALL, CONFIG)
    print(open(path).read())
    print(path)


if __name__ == '__main__':
    commandr.Run()
