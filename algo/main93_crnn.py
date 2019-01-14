# -*- coding: utf-8 -*-
from __future__ import print_function
import importlib
import time
import commandr
import yaml
import shutil
import numpy as np
import tensorflow as tf
from algo.lib.dataset import IndexIterator
from algo.lib.evaluate93 import basic_evaluate
from algo.model.const import *
from algo.model.train_config import TrainConfig
from algo.lib.common import print_evaluation, load_lookup_table2, tokenized_to_tid_list, tid_dropout
from algo.model.nn_config import BaseNNConfig
from algo.nn.base import BaseNNModel
from algo.nn.common import dense, cnn, rnn_cell
from algo.nn.common.common import add_gaussian_noise_layer, build_dropout_keep_prob
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


class NNModel(BaseNNModel):
    name = 'm93_crnn'

    def build_neural_network(self, lookup_table):
        test_mode = tf.placeholder(tf.int8, None, name=TEST_MODE)
        label_gold = tf.placeholder(tf.int32, [None, ], name=LABEL_GOLD)
        sample_weights = tf.placeholder(tf.float32, [None, ], name=SAMPLE_WEIGHTS)
        lookup_table = tf.Variable(
            lookup_table, dtype=tf.float32, name=LOOKUP_TABLE,
            trainable=self.config.embedding_trainable
        )
        dropout_keep_prob = build_dropout_keep_prob(keep_prob=self.config.dropout_keep_prob, test_mode=test_mode)

        tid_0 = tf.placeholder(tf.int32, [self.config.batch_size, self.config.seq_len], name=TID_0)
        seq_len_0 = tf.placeholder(tf.int32, [None, ], name=SEQ_LEN_0)

        tid_1 = tf.placeholder(tf.int32, [self.config.batch_size, self.config.seq_len], name=TID_1)
        seq_len_1 = tf.placeholder(tf.int32, [None, ], name=SEQ_LEN_1)

        tid_2 = tf.placeholder(tf.int32, [self.config.batch_size, self.config.seq_len], name=TID_2)
        seq_len_2 = tf.placeholder(tf.int32, [None, ], name=SEQ_LEN_2)

        embedded_0 = tf.nn.embedding_lookup(lookup_table, tid_0)
        embedded_0 = add_gaussian_noise_layer(embedded_0, stddev=self.config.embedding_noise_stddev, test_mode=test_mode)

        embedded_1 = tf.nn.embedding_lookup(lookup_table, tid_1)
        embedded_1 = add_gaussian_noise_layer(embedded_1, stddev=self.config.embedding_noise_stddev, test_mode=test_mode)

        embedded_2 = tf.nn.embedding_lookup(lookup_table, tid_2)
        embedded_2 = add_gaussian_noise_layer(embedded_2, stddev=self.config.embedding_noise_stddev, test_mode=test_mode)

        with tf.variable_scope("rnn_0") as scope:
            cnn_output = cnn.build(embedded_0, self.config.filter_num, self.config.kernel_size)
            _, last_state_0 = tf.nn.dynamic_rnn(
                rnn_cell.build_gru(self.config.rnn_dim, dropout_keep_prob=dropout_keep_prob),
                inputs=cnn_output, sequence_length=seq_len_0 - self.config.filter_num + 1, dtype=tf.float32
            )

        with tf.variable_scope("rnn_1") as scope:
            cnn_output = cnn.build(embedded_1, self.config.filter_num, self.config.kernel_size)
            _, last_state_1 = tf.nn.dynamic_rnn(
                rnn_cell.build_gru(self.config.rnn_dim, dropout_keep_prob=dropout_keep_prob),
                inputs=cnn_output, sequence_length=seq_len_1 - self.config.filter_num + 1, dtype=tf.float32
            )

        with tf.variable_scope("rnn_2") as scope:
            cnn_output = cnn.build(embedded_2, self.config.filter_num, self.config.kernel_size)
            _, last_state_2 = tf.nn.dynamic_rnn(
                rnn_cell.build_gru(self.config.rnn_dim, dropout_keep_prob=dropout_keep_prob),
                inputs=cnn_output, sequence_length=seq_len_2 - self.config.filter_num + 1, dtype=tf.float32
            )
        dense_input = tf.concat([last_state_0, last_state_1, last_state_2], axis=1, name=HIDDEN_FEAT)
        dense_input = tf.nn.dropout(dense_input, keep_prob=dropout_keep_prob)

        y, w, b = dense.build(dense_input, dim_output=self.config.output_dim, output_name=PROB_PREDICT)
        # 计算loss
        _loss_1 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            logits=y, labels=label_gold, weights=sample_weights))

        _loss_2 = tf.constant(0., dtype=tf.float32)
        if self.config.l2_reg_lambda is not None and self.config.l2_reg_lambda > 0:
            _loss_2 += self.config.l2_reg_lambda * tf.nn.l2_loss(w)
        loss = tf.add(_loss_1, _loss_2, name=LOSS)

        # 预测标签
        tf.cast(tf.argmax(y, 1), tf.int32, name=LABEL_PREDICT)

        # 统一的后处理
        self.build_optimizer(loss=loss)
        self.set_graph(graph=tf.get_default_graph())


def load_dataset(vocab_id_mapping, max_seq_len, with_label=True, label_version=None, text_version=None):
    def seq_to_len_list(seq_list):
        return list(map(len, seq_list))

    def zero_pad_seq_list(seq_list, seq_len):
        return list(map(lambda _seq: _seq + [0] * (seq_len - len(_seq)), seq_list))

    def trim_tid_list(tid_list, max_len):
        return list(map(lambda _seq: _seq[:max_len], tid_list))

    datasets = dict()
    for mode in [TRAIN, TEST]:
        tid_list_0 = tokenized_to_tid_list(load_tokenized_list(data_config.path(mode, TURN, '0.ek')), vocab_id_mapping)
        tid_list_0 = trim_tid_list(tid_list_0, max_seq_len)
        seq_len_0 = seq_to_len_list(tid_list_0)

        tid_list_1 = tokenized_to_tid_list(load_tokenized_list(data_config.path(mode, TURN, '1.ek')), vocab_id_mapping)
        tid_list_1 = trim_tid_list(tid_list_1, max_seq_len)
        seq_len_1 = seq_to_len_list(tid_list_1)

        tid_list_2 = tokenized_to_tid_list(load_tokenized_list(data_config.path(mode, TURN, '2.ek')), vocab_id_mapping)
        tid_list_2 = trim_tid_list(tid_list_2, max_seq_len)
        seq_len_2 = seq_to_len_list(tid_list_2)

        datasets[mode] = {
            TID_0: tid_list_0,
            TID_1: tid_list_1,
            TID_2: tid_list_2,
            SEQ_LEN_0: np.asarray(seq_len_0),
            SEQ_LEN_1: np.asarray(seq_len_1),
            SEQ_LEN_2: np.asarray(seq_len_2),
        }
        if with_label:
            label_path = data_config.path(mode, LABEL, label_version)
            label_list = load_label_list(label_path)
            datasets[mode][LABEL_GOLD] = np.asarray(label_list)

    for mode in [TRAIN, TEST]:
        for key in [TID_0, TID_1, TID_2]:
            datasets[mode][key] = np.asarray(zero_pad_seq_list(datasets[mode][key], max_seq_len))

    if with_label:
        output_dim = max(datasets[TRAIN][LABEL_GOLD]) + 1
        return datasets, output_dim
    else:
        return datasets


feed_key = {
    TRAIN: [TID_0, TID_1, TID_2, SEQ_LEN_0, SEQ_LEN_1, SEQ_LEN_2, LABEL_GOLD],
    TEST: [TID_0, TID_1, TID_2, SEQ_LEN_0, SEQ_LEN_1, SEQ_LEN_2]
}

fetch_key = {
    TRAIN: [OPTIMIZER, LOSS, LABEL_PREDICT],
    TEST: [LABEL_PREDICT, PROB_PREDICT, HIDDEN_FEAT]
}


@commandr.command
def train(text_version='ek', label_version=None, config_path='config93_naive.yaml'):
    """
    python -m algo.main93_v2 train
    python3 -m algo.main93_v2 train -c config_ntua93.yaml

    :param text_version: string
    :param label_version: string
    :param config_path: string
    :return:
    """
    config_data = yaml.load(open(config_path))

    output_key = '{}_{}_{}'.format(NNModel.name, text_version, int(time.time()))
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
    datasets, output_dim = load_dataset(
        vocab_id_mapping=vocab_id_mapping, max_seq_len=nn_config.seq_len,
        with_label=True, label_version=label_version, text_version=text_version
    )

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

    eval_history = {TRAIN: list(), DEV: list(), TEST: list()}

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

                if train_config.input_dropout_keep_prob < 1.:
                    for _key in [TID_0, TID_1, TID_2]:
                        var = nn.var(_key)
                        _tids = feed_dict[var]
                        feed_dict[var] = tid_dropout(_tids, train_config.input_dropout_keep_prob)

                res = sess.run(fetches=fetches[TRAIN], feed_dict=feed_dict)

                labels_predict += res[LABEL_PREDICT].tolist()
                labels_gold += dataset[LABEL_GOLD][batch_index].tolist()

            labels_predict, labels_gold = labels_predict[:n_sample], labels_gold[:n_sample]
            res = basic_evaluate(gold=labels_gold, pred=labels_predict)
            print_evaluation(res)
            eval_history[TRAIN].append(res)

            global_step = tf.train.global_step(sess, nn.var(GLOBAL_STEP))

            if train_config.valid_rate == 0.:
                if best_res[TRAIN] is None or res[early_stop_metric] > best_res[TRAIN][early_stop_metric]:
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
                eval_history[DEV].append(res)
                print_evaluation(res)

                # Early Stop
                if best_res[VALID] is None or res[early_stop_metric] > best_res[VALID][early_stop_metric]:
                    saver.save(sess, save_path=model_output_prefix, global_step=global_step)
                    best_res[VALID] = res
                    no_update_count[VALID] = 0
                else:
                    no_update_count[VALID] += 1

            # eval test
            _mode = TEST
            _dataset = datasets[_mode]
            _index_iterator = index_iterators[_mode]
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

            if no_update_count[TRAIN] >= max_no_update_count:
                break

        # 训练结束 ##########################################################################
        # 确保输出文件夹存在

    print('========================= BEST ROUND EVALUATION =========================')

    json.dump(eval_history, open(data_config.output_path(output_key, 'eval', 'json'), 'w'))

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

    test_score_list = map(lambda _item: _item['f1'], eval_history[TEST])
    print('best test f1 reached: {}'.format(max(test_score_list)))

    print('OUTPUT_KEY: {}'.format(output_key))


if __name__ == '__main__':
    commandr.Run()
