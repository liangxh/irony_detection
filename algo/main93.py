# -*- coding: utf-8 -*-
from __future__ import print_function
import importlib
import time
import commandr
import yaml
import shutil
import numpy as np
import tensorflow as tf
from collections import defaultdict
from algo.nn.base import BaseNNModel
from algo.lib.dataset import IndexIterator, SimpleIndexIterator
from algo.lib.evaluate93 import basic_evaluate
from algo.model.const import *
from algo.model.train_config import TrainConfig
from dataset.common.const import *
from dataset.common.load import *
from algo.lib.common import print_evaluation, load_lookup_table, tokenized_to_tid_list, build_random_lookup_table
from dataset.semeval2019_task3_dev.config import config as data_config
from dataset.semeval2019_task3_dev.process import Processor, label_str

MAX_WORD_SEQ_LEN = 170
MAX_CHAR_SEQ_LEN = 170
CHAR = 'char'
WORD = 'word'


def load_dataset(data_config, analyzer, vocab_id_mapping, seq_len, with_label=True, label_version=None, text_version=None):
    def seq_to_len_list(seq_list):
        return list(map(len, seq_list))

    def zero_pad_seq_list(seq_list, seq_len):
        return list(map(lambda _seq: _seq + [0] * (seq_len - len(_seq)), seq_list))

    datasets = dict()
    for mode in [TRAIN, TEST]:
        if analyzer == WORD:
            text_path = data_config.path(mode, TEXT, text_version)
            tokenized_list = load_tokenized_list(text_path)
        elif analyzer == CHAR:
            text_path = data_config.path(mode, TEXT)
            text_list = load_text_list(text_path)
            tokenized_list = list(map(list, text_list))
        else:
            raise ValueError('invalid analyzer, got {}'.format(analyzer))

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
    TEST: [LABEL_PREDICT, PROB_PREDICT, HIDDEN_FEAT]
}


@commandr.command
def train(dataset_key, text_version, label_version=None, config_path='config.yaml'):
    """
    python -m algo.main93 train semeval2019_task3_dev -t ek
    python3 -m algo.main93 train semeval2019_task3_dev -t ek -c config_ntua93.yaml

    :param dataset_key: string
    :param text_version: string
    :param label_version: string
    :param config_path: string
    :return:
    """
    config_data = yaml.load(open(config_path))

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

    if config_data['analyzer'] == WORD:
        w2v_key = '{}_{}'.format(config_data['word']['w2v_version'], text_version)
        w2v_model_path = data_config.path(ALL, WORD2VEC, w2v_key)
        vocab_train_path = data_config.path(TRAIN, VOCAB, text_version)

        # 加载字典集
        # 在模型中会采用所有模型中支持的词向量, 并为有足够出现次数的单词随机生成词向量
        vocab_meta_list = load_vocab_list(vocab_train_path)
        # vocab_meta_list += load_vocab_list(semeval2018_task3_date_config.path(TRAIN, VOCAB, text_version))
        vocabs = [_meta['t'] for _meta in vocab_meta_list if _meta['tf'] >= config_data[WORD]['min_tf']]

        # 加载词向量与相关数据
        lookup_table, vocab_id_mapping, embedding_dim = load_lookup_table(
            w2v_model_path=w2v_model_path, vocabs=vocabs)
        json.dump(vocab_id_mapping, open(data_config.output_path(output_key, ALL, VOCAB_ID_MAPPING), 'w'))
        max_seq_len = MAX_WORD_SEQ_LEN
    elif config_data['analyzer'] == CHAR:
        texts = load_text_list(data_config.path(TRAIN, TEXT))
        char_set = set()
        for text in texts:
            char_set |= set(text)
        lookup_table, vocab_id_mapping, embedding_dim = build_random_lookup_table(
            vocabs=char_set, dim=config_data['char']['embedding_dim'])
        max_seq_len = MAX_CHAR_SEQ_LEN
    else:
        raise ValueError('invalid analyzer: {}'.format(config_data['analyzer']))

    # 加载训练数据
    datasets, output_dim = load_dataset(
        data_config=data_config, analyzer=config_data['analyzer'],
        vocab_id_mapping=vocab_id_mapping, seq_len=max_seq_len,
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
    nn_config.set_seq_len(max_seq_len)
    # 搭建神经网络
    nn = NNModel(config=nn_config)
    nn.build_neural_network(lookup_table=lookup_table)

    batch_size = train_config.batch_size
    fetches = {mode: {_key: nn.var(_key) for _key in fetch_key[mode]} for mode in [TRAIN, TEST]}
    last_eval = {TRAIN: None, VALID: None, TEST: None}

    model_output_prefix = data_config.model_path(key=output_key) + '/model'

    best_res = {mode: None for mode in [TRAIN, VALID]}
    no_update_count = {mode: 0 for mode in [TRAIN, VALID]}
    max_no_update_count = 10

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
                feed_dict[nn.var(SAMPLE_WEIGHTS)] = list(map(label_weight.get, feed_dict[nn.var(LABEL_GOLD)]))
                feed_dict[nn.var(TEST_MODE)] = 0
                res = sess.run(fetches=fetches[TRAIN], feed_dict=feed_dict)

                labels_predict += res[LABEL_PREDICT].tolist()
                labels_gold += dataset[LABEL_GOLD][batch_index].tolist()

            labels_predict, labels_gold = labels_predict[:n_sample], labels_gold[:n_sample]
            labels_predict, labels_gold = labels_predict[:n_sample], labels_gold[:n_sample]
            res = basic_evaluate(gold=labels_gold, pred=labels_predict)
            last_eval[TRAIN] = res
            print_evaluation(res)

            global_step = tf.train.global_step(sess, nn.var(GLOBAL_STEP))

            if train_config.valid_rate == 0.:
                if best_res[TRAIN] is None or res[F1_SCORE] > best_res[TRAIN][F1_SCORE]:
                    best_res[TRAIN] = res
                    no_update_count[TRAIN] = 0
                    saver.save(sess, save_path=model_output_prefix, global_step=global_step)
                else:
                    no_update_count[TRAIN] += 1
            else:
                if best_res[TRAIN] is None or res[F1_SCORE] > best_res[TRAIN][F1_SCORE]:
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
                last_eval[VALID] = res
                print_evaluation(res)

                # Early Stop
                if best_res[VALID] is None or res[F1_SCORE] > best_res[VALID][F1_SCORE]:
                    saver.save(sess, save_path=model_output_prefix, global_step=global_step)
                    best_res[VALID] = res
                    no_update_count[VALID] = 0
                else:
                    no_update_count[VALID] += 1

            if no_update_count[TRAIN] >= max_no_update_count:
                break

        # 训练结束 ##########################################################################
        # 确保输出文件夹存在

    print('========================= BEST ROUND EVALUATION =========================')

    with tf.Session() as sess:
        prefix_checkpoint = tf.train.latest_checkpoint(data_config.model_path(key=output_key))
        saver = tf.train.import_meta_graph('{}.meta'.format(prefix_checkpoint))
        saver.restore(sess, prefix_checkpoint)

        nn = BaseNNModel(config=None)
        nn.set_graph(tf.get_default_graph())

        for mode in [TRAIN, TEST]:
            dataset = datasets[mode]
            index_iterator = index_iterators[mode]
            n_sample = index_iterator.n_sample()

            prob_predict = list()
            labels_predict = list()
            labels_gold = list()
            hidden_feats = list()

            for batch_index in index_iterator.iterate(batch_size, shuffle=False):
                feed_dict = {nn.var(_key): dataset[_key][batch_index] for _key in feed_key[TEST]}
                feed_dict[nn.var(TEST_MODE)] = 1
                res = sess.run(fetches=fetches[TEST], feed_dict=feed_dict)
                prob_predict += res[PROB_PREDICT].tolist()
                labels_predict += res[LABEL_PREDICT].tolist()
                hidden_feats += res[HIDDEN_FEAT].tolist()
                labels_gold += dataset[LABEL_GOLD][batch_index].tolist()

            prob_predict = prob_predict[:n_sample]
            labels_predict = labels_predict[:n_sample]
            labels_gold = labels_gold[:n_sample]
            hidden_feats = hidden_feats[:n_sample]

            if mode == TEST:
                res = basic_evaluate(gold=labels_gold, pred=labels_predict)
                best_res[TEST] = res

            # 导出隐藏层
            with open(data_config.output_path(output_key, mode, HIDDEN_FEAT), 'w') as file_obj:
                for _feat in hidden_feats:
                    file_obj.write('\t'.join(map(str, _feat)) + '\n')
            # 导出预测的label
            with open(data_config.output_path(output_key, mode, LABEL_PREDICT), 'w') as file_obj:
                for _label in labels_predict:
                    file_obj.write('{}\n'.format(_label))
            with open(data_config.output_path(output_key, mode, PROB_PREDICT), 'w') as file_obj:
                for _prob in prob_predict:
                    file_obj.write('\t'.join(map(str, _prob)) + '\n')

        for mode in [TRAIN, VALID, TEST]:
            if mode == VALID and train_config.valid_rate == 0.:
                continue
            res = best_res[mode]
            print(mode)
            print_evaluation(res)

            json.dump(res, open(data_config.output_path(output_key, mode, EVALUATION), 'w'))
            print()

    print('OUTPUT_KEY: {}'.format(output_key))


@commandr.command('eval')
def show_eval(output_key):
    """
    [Usage]
    python algo/main.py eval A_ntua_ek_1542454066

    :param output_key: string
    :return:
    """
    for mode in [TRAIN, VALID, TEST]:
        res = json.load(open(data_config.output_path(output_key, mode, EVALUATION)))
        print(mode)
        print_evaluation(res)
        for col in res[CONFUSION_MATRIX]:
            print(','.join(map(str, col)))
        print()


@commandr.command('eval2')
def show_eval2(output_key):
    for mode in [TRAIN, TEST]:
        path_predict = data_config.output_path(output_key, mode, LABEL_PREDICT)
        labels_predict = load_label_list(path_predict)

        path_gold = data_config.path(mode, LABEL)
        labels_gold = load_label_list(path_gold)

        res = basic_evaluate(gold=labels_gold, pred=labels_predict)
        print(mode)
        print_evaluation(res)
        for col in res[CONFUSION_MATRIX]:
            print(','.join(map(str, col)))
        print()


@commandr.command('clear')
def clear_output(output_key):
    """
    [Usage]
    python algo/main.py clear A_ntua_ek_1542595525
    python3 -m algo.main93 clear xxxxxxx

    :param output_key: string
    :return:
    """
    shutil.rmtree(data_config.output_folder(output_key))
    shutil.rmtree(data_config.model_folder(output_key))


@commandr.command('config')
def show_config(output_key):
    """
    [Usage]
    python3 -m algo.main93 config xxxxxx

    :param output_key:
    :return:
    """
    path = data_config.output_path(output_key, ALL, CONFIG)
    print(open(path).read())
    print(path)


@commandr.command
def check_seq(output_key, w2v_key='ntua_ek'):
    mode = TEST
    path = data_config.output_path(output_key, mode, LABEL_PREDICT)
    pred = load_label_list(path)

    path = data_config.path(mode, LABEL)
    gold = load_label_list(path)

    w2v_model_path = data_config.path(ALL, WORD2VEC, w2v_key)
    vocab_train_path = data_config.path(TRAIN, VOCAB, 'ek')

    # 加载字典集
    # 在模型中会采用所有模型中支持的词向量, 并为有足够出现次数的单词随机生成词向量
    vocab_meta_list = load_vocab_list(vocab_train_path)
    vocabs = [_meta['t'] for _meta in vocab_meta_list if _meta['tf'] >= 2]

    # 加载词向量与相关数据
    lookup_table, vocab_id_mapping, embedding_dim = load_lookup_table(
        w2v_model_path=w2v_model_path, vocabs=vocabs)

    tokens_0 = load_tokenized_list(data_config.path(mode, TURN, '0.ek'))
    tokens_1 = load_tokenized_list(data_config.path(mode, TURN, '1.ek'))
    tokens_2 = load_tokenized_list(data_config.path(mode, TURN, '2.ek'))
    tid_list_0 = tokenized_to_tid_list(tokens_0, vocab_id_mapping)
    tid_list_1 = tokenized_to_tid_list(tokens_1, vocab_id_mapping)
    tid_list_2 = tokenized_to_tid_list(load_tokenized_list(data_config.path(mode, TURN, '2.ek')), vocab_id_mapping)

    max_seq_len = 0
    for p, g, tid_0, tid_1, tid_2, tk_0, tk_1, tk_2 in zip(pred, gold, tid_list_0, tid_list_1, tid_list_2, tokens_0, tokens_1, tokens_2):
        if p != g and (len(tid_0) > 30 or len(tid_1) > 30 or len(tid_2) > 30):
            print('pred: {}, gold: {}'.format(p, g))
            print('turn0: {}'.format(' '.join(tk_0)))
            print('turn1: {}'.format(' '.join(tk_1)))
            print('turn2: {}'.format(' '.join(tk_2)))

        if p != g:
            max_seq_len = max(max_seq_len, len(tid_0), len(tid_1), len(tid_2))
    print(max_seq_len)


@commandr.command
def check_wrong(output_key, w2v_key='ntua_ek'):
    mode = TEST
    path = data_config.output_path(output_key, mode, LABEL_PREDICT)
    pred = load_label_list(path)

    path = data_config.path(mode, LABEL)
    gold = load_label_list(path)

    w2v_model_path = data_config.path(ALL, WORD2VEC, w2v_key)
    vocab_train_path = data_config.path(TRAIN, VOCAB, 'ek')

    # 加载字典集
    # 在模型中会采用所有模型中支持的词向量, 并为有足够出现次数的单词随机生成词向量
    vocab_meta_list = load_vocab_list(vocab_train_path)
    vocabs = [_meta['t'] for _meta in vocab_meta_list if _meta['tf'] >= 2]

    # 加载词向量与相关数据
    lookup_table, vocab_id_mapping, embedding_dim = load_lookup_table(
        w2v_model_path=w2v_model_path, vocabs=vocabs)

    tokens_0 = load_tokenized_list(data_config.path(mode, TURN, '0.ek'))
    tokens_1 = load_tokenized_list(data_config.path(mode, TURN, '1.ek'))
    tokens_2 = load_tokenized_list(data_config.path(mode, TURN, '2.ek'))
    tid_list_0 = tokenized_to_tid_list(tokens_0, vocab_id_mapping)
    tid_list_1 = tokenized_to_tid_list(tokens_1, vocab_id_mapping)
    tid_list_2 = tokenized_to_tid_list(load_tokenized_list(data_config.path(mode, TURN, '2.ek')), vocab_id_mapping)

    max_seq_len = 0
    for p, g, tid_0, tid_1, tid_2, tk_0, tk_1, tk_2 in zip(pred, gold, tid_list_0, tid_list_1, tid_list_2, tokens_0, tokens_1, tokens_2):
        if p != g and (len(tid_0) > 30 or len(tid_1) > 30 or len(tid_2) > 30):
            print('pred: {}, gold: {}'.format(p, g))
            print('turn0: {}'.format(' '.join(tk_0)))
            print('turn1: {}'.format(' '.join(tk_1)))
            print('turn2: {}'.format(' '.join(tk_2)))

        if p != g:
            max_seq_len = max(max_seq_len, len(tid_0), len(tid_1), len(tid_2))
    print(max_seq_len)


@commandr.command
def export_wrong(output_key):
    mode = TEST
    path = data_config.output_path(output_key, mode, LABEL_PREDICT)
    pred = load_label_list(path)

    path = data_config.path(mode, LABEL)
    gold = load_label_list(path)

    tokens_0 = load_tokenized_list(data_config.path(mode, TURN, '0.ek'))
    tokens_1 = load_tokenized_list(data_config.path(mode, TURN, '1.ek'))
    tokens_2 = load_tokenized_list(data_config.path(mode, TURN, '2.ek'))

    wrong = defaultdict(lambda: defaultdict(lambda: list()))

    max_seq_len = 0
    for p, g, tk_0, tk_1, tk_2 in zip(pred, gold, tokens_0, tokens_1, tokens_2):
        if p != g:
            wrong[g][p].append(' '.join(tk_0) + ' | ' + ' '.join(tk_1) + ' | ' + ' '.join(tk_2))
            max_seq_len = max(max_seq_len, len(tk_0), len(tk_1), len(tk_2))

    for _g in range(4):
        for _p in range(4):
            print('{}->{}'.format(_g, _p))
            for sample in wrong[_g][_p]:
                print('\t{}'.format(sample))

    print(max_seq_len)


@commandr.command
def export_error(filename):
    dataset = Processor.load_origin(filename)

    path = data_config.path(FINAL, LABEL)
    gold = load_label_list(path)
    wrong = defaultdict(lambda: defaultdict(lambda: list()))

    for g, sample in zip(gold, dataset):
        p = sample[-1]
        if p != g:
            wrong[g][p].append(sample[0] + ' | ' + sample[1] + ' | ' + sample[2])

    for _g in range(4):
        for _p in range(4):
            print('{}->{}'.format(label_str[_g], label_str[_p]))
            for sample in wrong[_g][_p]:
                print('\t{}'.format(sample))


if __name__ == '__main__':
    commandr.Run()
