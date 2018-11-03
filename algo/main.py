# -*- coding: utf-8 -*-
from __future__ import print_function
import importlib
import time
import commandr
import yaml
import shutil
import numpy as np
import tensorflow as tf
from algo.nn.base import BaseNNModel
from algo.lib.dataset import IndexIterator, SimpleIndexIterator
from algo.lib.evaluate import basic_evaluate
from algo.model.const import *
from algo.model.train_config import TrainConfig
from dataset.common.const import *
from dataset.common.load import *
from algo.lib.common import print_evaluation, load_lookup_table, tokenized_to_tid_list
from dataset.semeval2018_task3.config import config as semeval2018_task3_date_config

MAX_SEQ_LEN = 90


def load_dataset(data_config, vocab_id_mapping, seq_len, with_label=True, label_version=None, text_version=TEXT):
    def seq_to_len_list(seq_list):
        return map(len, seq_list)

    def zero_pad_seq_list(seq_list, seq_len):
        return map(lambda _seq: _seq + [0] * (seq_len - len(_seq)), seq_list)

    datasets = dict()
    for mode in [TRAIN, TEST]:
        text_path = data_config.path(mode, TEXT, text_version)
        tokenized_list = load_tokenized_list(text_path)

        tid_list = tokenized_to_tid_list(tokenized_list, vocab_id_mapping)
        seq_len_list = seq_to_len_list(tid_list)
        datasets[mode] = {
            TOKEN_ID_SEQ: tid_list,
            SEQ_LEN: np.asarray(seq_len_list),
        }
        if with_label:
            label_path = data_config.path(mode, LABEL, label_version)
            label_list = load_label_list(label_path)
            datasets[mode][LABEL_GOLD] = np.asarray(label_list)

    max_seq_len = -1
    for _dataset in datasets.values():
        max_seq_len = max(max_seq_len, _dataset[SEQ_LEN].max() + 1)

    if seq_len < max_seq_len:
        raise ValueError('seq_len set as {}, got max seq_len = {}'.format(seq_len, max_seq_len))

    for mode in [TRAIN, TEST]:
        datasets[mode][TOKEN_ID_SEQ] = np.asarray(zero_pad_seq_list(datasets[mode][TOKEN_ID_SEQ], seq_len))

    if with_label:
        output_dim = max(datasets[TRAIN][LABEL_GOLD]) + 1
        return datasets, output_dim
    else:
        return datasets


feed_key = {
    TRAIN: [TOKEN_ID_SEQ, SEQ_LEN, LABEL_GOLD],
    TEST: [TOKEN_ID_SEQ, SEQ_LEN]
}

fetch_key = {
    TRAIN: [OPTIMIZER, LOSS, LABEL_PREDICT],
    TEST: [LABEL_PREDICT, HIDDEN_FEAT]
}


@commandr.command
def train(dataset_key, text_version=TEXT, label_version=None, config_path='config.yaml'):
    """
    python algo/main.py train semeval2018_task3 -l A -t ek
    python algo/main.py train semeval2018_task1 -l love
    python algo/main.py train semeval2014_task9

    :param dataset_key: string
    :param text_version: string
    :param label_version: string
    :param config_path: string
    :return:
    """
    pos_label = None
    if dataset_key == 'semeval2018_task3' and label_version == 'A':
        pos_label = 1

    config_data = yaml.load(open(config_path))

    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')

    w2v_model_path = data_config.path(ALL, WORD2VEC, config_data['version']['w2v'].format(text_version=text_version))
    vocab_train_path = data_config.path(TRAIN, VOCAB, text_version)

    output_key = '{}_{}_{}'.format(config_data['module'].rsplit('.', 1)[1], text_version, int(time.time()))
    if label_version is not None:
        output_key = '{}_{}'.format(label_version, output_key)
    print('OUTPUT_KEY: {}'.format(output_key))

    # 准备输出路径的文件夹
    data_config.prepare_output_folder(output_key=output_key)
    data_config.prepare_model_folder(output_key=output_key)

    shutil.copy(config_path, data_config.output_path(output_key, ALL, CONFIG))

    # 根据配置加载模块
    module_relative_path = config_data['module']
    NNModel = getattr(importlib.import_module(module_relative_path), 'NNModel')
    NNConfig = getattr(importlib.import_module(module_relative_path), 'NNConfig')

    # 加载字典集
    # 在模型中会采用所有模型中支持的词向量, 并为有足够出现次数的单词随机生成词向量
    vocab_meta_list = load_vocab_list(vocab_train_path)
    vocab_meta_list += load_vocab_list(semeval2018_task3_date_config.path(TRAIN, VOCAB, text_version))
    vocabs = [_meta['t'] for _meta in vocab_meta_list if _meta['tf'] >= 2]

    # 加载词向量与相关数据
    lookup_table, vocab_id_mapping, embedding_dim = load_lookup_table(
        w2v_model_path=w2v_model_path, vocabs=vocabs)
    json.dump(vocab_id_mapping, open(data_config.output_path(output_key, ALL, VOCAB_ID_MAPPING), 'w'))

    # 加载训练数据
    datasets, output_dim = load_dataset(
        data_config=data_config, vocab_id_mapping=vocab_id_mapping, seq_len=MAX_SEQ_LEN,
        with_label=True, label_version=label_version, text_version=text_version)

    # 加载配置
    nn_config = NNConfig(config_data)
    train_config = TrainConfig(config_data['train'])

    # 初始化数据集的检索
    index_iterators = {mode: IndexIterator(datasets[mode][LABEL_GOLD]) for mode in [TRAIN, TEST]}
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
    nn_config.set_seq_len(MAX_SEQ_LEN)
    # 搭建神经网络
    nn = NNModel(config=nn_config)
    nn.build_neural_network(lookup_table=lookup_table)

    batch_size = train_config.batch_size
    fetches = {mode: {_key: nn.var(_key) for _key in fetch_key[mode]} for mode in [TRAIN, TEST]}
    last_eval = {TRAIN: None, VALID: None, TEST: None}

    model_output_prefix = data_config.model_path(key=output_key) + '/model'

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        dataset = datasets[TRAIN]
        index_iterator = index_iterators[TRAIN]

        # 训练开始 ##########################################################################
        for epoch in range(train_config.epoch):
            print('== epoch {} =='.format(epoch))

            # 利用训练集进行训练
            print('TRAIN')
            n_sample = index_iterator.n_sample(TRAIN)
            labels_predict = list()
            labels_gold = list()

            for batch_index in index_iterator.iterate(batch_size, mode=TRAIN, shuffle=True):
                feed_dict = {nn.var(_key): dataset[_key][batch_index] for _key in feed_key[TRAIN]}
                feed_dict[nn.var(DROPOUT_KEEP_PROB)] = train_config.dropout_keep_prob
                feed_dict[nn.var(SAMPLE_WEIGHTS)] = map(label_weight.get, feed_dict[nn.var(LABEL_GOLD)])
                res = sess.run(fetches=fetches[TRAIN], feed_dict=feed_dict)

                labels_predict += res[LABEL_PREDICT].tolist()
                labels_gold += dataset[LABEL_GOLD][batch_index].tolist()

            labels_predict, labels_gold = labels_predict[:n_sample], labels_gold[:n_sample]
            labels_predict, labels_gold = labels_predict[:n_sample], labels_gold[:n_sample]
            res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)
            last_eval[TRAIN] = res
            print_evaluation(res)

            global_step = tf.train.global_step(sess, nn.var(GLOBAL_STEP))
            saver.save(sess, save_path=model_output_prefix, global_step=global_step)

            if train_config.valid_rate > 0.:
                # 计算在验证集上的表现, 不更新模型参数
                print('VALID')
                n_sample = index_iterator.n_sample(VALID)
                labels_predict = list()
                labels_gold = list()

                for batch_index in index_iterator.iterate(batch_size, mode=VALID, shuffle=False):
                    feed_dict = {nn.var(_key): dataset[_key][batch_index] for _key in feed_key[TEST]}
                    feed_dict[nn.var(DROPOUT_KEEP_PROB)] = 1.
                    res = sess.run(fetches=fetches[TEST], feed_dict=feed_dict)
                    labels_predict += res[LABEL_PREDICT].tolist()
                    labels_gold += dataset[LABEL_GOLD][batch_index].tolist()

                labels_predict, labels_gold = labels_predict[:n_sample], labels_gold[:n_sample]
                res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)
                last_eval[VALID] = res
                print_evaluation(res)

        # 训练结束 ##########################################################################
        # 确保输出文件夹存在

        for mode in [TRAIN, TEST]:
            dataset = datasets[mode]
            index_iterator = index_iterators[mode]
            n_sample = index_iterator.n_sample()

            labels_predict = list()
            labels_gold = list()
            hidden_feats = list()
            for batch_index in index_iterator.iterate(batch_size, shuffle=False):
                feed_dict = {nn.var(_key): dataset[_key][batch_index] for _key in feed_key[TEST]}
                feed_dict[nn.var(DROPOUT_KEEP_PROB)] = 1.
                res = sess.run(fetches=fetches[TEST], feed_dict=feed_dict)
                labels_predict += res[LABEL_PREDICT].tolist()
                hidden_feats += res[HIDDEN_FEAT].tolist()
                labels_gold += dataset[LABEL_GOLD][batch_index].tolist()

            labels_predict = labels_predict[:n_sample]
            labels_gold = labels_gold[:n_sample]
            hidden_feats = hidden_feats[:n_sample]

            if mode == TEST:
                res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)
                last_eval[TEST] = res

            # 导出隐藏层
            with open(data_config.output_path(output_key, mode, HIDDEN_FEAT), 'w') as file_obj:
                for _feat in hidden_feats:
                    file_obj.write('\t'.join(map(str, _feat)) + '\n')
            # 导出预测的label
            with open(data_config.output_path(output_key, mode, LABEL_PREDICT), 'w') as file_obj:
                for _label in labels_predict:
                    file_obj.write('{}\n'.format(_label))

        print('========================= FINAL EVALUATION =========================')
        for mode in [TRAIN, VALID, TEST]:
            if mode == VALID and train_config.valid_rate == 0.:
                continue
            res = last_eval[mode]
            print(mode)
            print_evaluation(res)

            json.dump(res, open(data_config.output_path(output_key, mode, EVALUATION), 'w'))
            print()

    print('OUTPUT_KEY: {}'.format(output_key))


@commandr.command('feat')
def build_feat(dataset_key_src, output_key_src, dataset_key_dest='semeval2018_task3', text_version=TEXT):
    """
    python algo/main.py feat semeval2014_task9 gru_1539175546
    python algo/main.py feat semeval2018_task1 love_gru_1539178720

    :param dataset_key_src: 模型对应的dataset_key
    :param output_key_src: 模型对应的output_key
    :param dataset_key_dest: 需要生成特征向量的数据集对应的dataset_key
    :param text_version:
    :return:
    """
    output_key = '{}.{}'.format(dataset_key_src, output_key_src)
    print('OUTPUT_KEY: {}'.format(output_key))

    # 获取模型文件所在路径
    data_src_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key_src)), 'config')
    model_output_prefix = data_src_config.model_path(key=output_key_src)
    # 加载模型对应字典
    vocab_id_mapping = json.load(open(data_src_config.output_path(output_key_src, ALL, VOCAB_ID_MAPPING), 'r'))

    # 加载训练数据
    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key_dest)), 'config')
    data_config.prepare_output_folder(output_key=output_key)
    datasets = load_dataset(
        data_config=data_config, vocab_id_mapping=vocab_id_mapping, seq_len=MAX_SEQ_LEN,
        with_label=False, text_version=text_version
    )
    batch_size = 200

    with tf.Session() as sess:
        prefix_checkpoint = tf.train.latest_checkpoint(model_output_prefix)
        print(prefix_checkpoint)
        saver = tf.train.import_meta_graph('{}.meta'.format(prefix_checkpoint))
        saver.restore(sess, prefix_checkpoint)

        nn = BaseNNModel(config=None)
        nn.set_graph(tf.get_default_graph())
        fetches = {mode: {_key: nn.var(_key) for _key in [HIDDEN_FEAT, ]} for mode in [TEST, ]}

        for mode in [TRAIN, TEST]:
            dataset = datasets[mode]
            index_iterator = SimpleIndexIterator(n_sample=dataset[SEQ_LEN].shape[0])
            n_sample = index_iterator.n_sample()

            hidden_feats = list()
            for batch_index in index_iterator.iterate(batch_size):
                feed_dict = {nn.var(_key): dataset[_key][batch_index] for _key in feed_key[TEST]}
                feed_dict[nn.var(DROPOUT_KEEP_PROB)] = 1.
                res = sess.run(fetches=fetches[TEST], feed_dict=feed_dict)
                hidden_feats += res[HIDDEN_FEAT].tolist()
            hidden_feats = hidden_feats[:n_sample]

            # 导出隐藏层
            with open(data_config.output_path(output_key, mode, HIDDEN_FEAT), 'w') as file_obj:
                for _feat in hidden_feats:
                    file_obj.write('\t'.join(map(str, _feat)) + '\n')

    print('OUTPUT_KEY: {}'.format(output_key))


@commandr.command('eval')
def show_eval(dataset_key, output_key):
    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')

    for mode in [TRAIN, VALID, TEST]:
        res = json.load(open(data_config.output_path(output_key, mode, EVALUATION)))
        print(mode)
        print_evaluation(res)


if __name__ == '__main__':
    commandr.Run()
