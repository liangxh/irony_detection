# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import time
import copy
import commandr
import yaml
import random
import shutil
import numpy as np
import tensorflow as tf
from algo.lib.dataset import IndexIterator, SimpleIndexIterator
from algo.lib.evaluate import basic_evaluate
from algo.model.const import *
from algo.model.train_config import TrainConfig
from algo.lib.common import print_evaluation, load_lookup_table2, tokenized_to_tid_list
from algo.model.nn_config import BaseNNConfig
from algo.nn.base import BaseNNModel
from algo.nn.common import dense, cnn, rnn_cell, attention
from algo.nn.common.common import add_gaussian_noise_layer, build_dropout_keep_prob, mask_by_seq_len
from dataset.common.const import *
from dataset.common.load import *
from dataset.semeval2018_task3.config import config as data_config


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
    name = 'blstm'

    def build_neural_network(self, lookup_table):
        test_mode = tf.placeholder(tf.int8, None, name=TEST_MODE)
        label_gold = tf.placeholder(tf.int32, [None, ], name=LABEL_GOLD)
        sample_weights = tf.placeholder(tf.float32, [None, ], name=SAMPLE_WEIGHTS)
        lookup_table = tf.Variable(
            lookup_table, dtype=tf.float32, name=LOOKUP_TABLE,
            trainable=self.config.embedding_trainable
        )
        dropout_keep_prob = build_dropout_keep_prob(keep_prob=self.config.dropout_keep_prob, test_mode=test_mode)

        tid = tf.placeholder(tf.int32, [self.config.batch_size, self.config.seq_len], name=TID)
        seq_len = tf.placeholder(tf.int32, [None, ], name=SEQ_LEN)
        embedded = tf.nn.embedding_lookup(lookup_table, tid)

        if self.config.embedding_noise_type is None:
            pass
        elif self.config.embedding_noise_type == 'gaussian':
            embedded = add_gaussian_noise_layer(
                embedded, stddev=self.config.embedding_noise_stddev, test_mode=test_mode)
        elif self.config.embedding_noise_type == 'dropout':
            emb_dropout_keep_prob = build_dropout_keep_prob(
                keep_prob=self.config.embedding_dropout_keep_prob, test_mode=test_mode)
            embedded = tf.nn.dropout(embedded, emb_dropout_keep_prob)
        else:
            raise Exception('unknown embedding noise type: {}'.format(self.config.embedding_noise_type))

        with tf.variable_scope('bgru'):
            cell_fw = rnn_cell.build_lstm(self.config.rnn_dim, dropout_keep_prob=dropout_keep_prob)
            cell_bw = rnn_cell.build_lstm(self.config.rnn_dim, dropout_keep_prob=dropout_keep_prob)

            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, embedded, seq_len,
                cell_fw.zero_state(self.config.batch_size, tf.float32),
                cell_bw.zero_state(self.config.batch_size, tf.float32)
            )
            outputs = tf.concat(outputs, axis=-1)
            last_state = tf.concat([states[0] for states in output_states], axis=-1)

        if self.config.use_attention:
            last_state, _ = attention.build(outputs, self.config.attention_dim)

        dense_input = tf.concat([last_state, ], axis=1, name=HIDDEN_FEAT)
        dense_input = tf.nn.dropout(dense_input, keep_prob=dropout_keep_prob)

        l2_component = None
        for conf in self.config.dense_layers:
            dense_input, w, _ = dense.build(
                dense_input, dim_output=conf['dim'], activation=getattr(tf.nn, conf['activation']))
            if conf.get('l2', 0.) > 0:
                comp = conf['l2'] * tf.nn.l2_loss(w)
                if l2_component is None:
                    l2_component = comp
                else:
                    l2_component += comp

        l2_w_list = list()
        if self.config.max_out is None:
            y, w, b = dense.build(dense_input, dim_output=self.config.output_dim, output_name=PROB_PREDICT)
            l2_w_list.append(w)
        else:
            y_list = list()
            for dim in self.config.max_out:
                y, w, b = dense.build(dense_input, dim_output=dim)
                y = tf.expand_dims(tf.reduce_max(y, 1), axis=1)
                y_list.append(y)
                l2_w_list.append(w)

            y = tf.concat(y_list, axis=1, name=PROB_PREDICT)

        # 计算loss
        _loss_1 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            logits=y, labels=label_gold, weights=sample_weights))

        _loss_2 = tf.constant(0., dtype=tf.float32)
        if self.config.l2_reg_lambda is not None and self.config.l2_reg_lambda > 0:
            for w in l2_w_list:
                _loss_2 += self.config.l2_reg_lambda * tf.nn.l2_loss(w)
        if l2_component is not None:
            _loss_2 += l2_component
        loss = tf.add(_loss_1, _loss_2, name=LOSS)

        # 预测标签
        tf.cast(tf.argmax(y, 1), tf.int32, name=LABEL_PREDICT)

        # 统一的后处理
        self.build_optimizer(loss=loss)
        self.set_graph(graph=tf.get_default_graph())


def to_nn_input(tid_list, max_seq_len):
    tid_list = trim_tid_list(tid_list, max_seq_len)
    seq_len = np.asarray(list(map(lambda _seq: min(max_seq_len, len(_seq) + 1), tid_list)))
    tid_list = np.asarray(zero_pad_seq_list(tid_list, max_seq_len))
    return tid_list, seq_len


def load_dataset(mode, vocab_id_mapping, max_seq_len, sampling=False, with_label=True, label_version=None):
    dataset = dict()
    tid_list = tokenized_to_tid_list(
        load_tokenized_list(data_config.path(mode, TEXT, EK)),
        vocab_id_mapping
    )
    dataset[TID] = tid_list
    print('{}: {}'.format(mode, max(list(map(lambda _item: len(_item), tid_list)))))

    if with_label:
        label_path = data_config.path(mode, LABEL, label_version)
        label_list = load_label_list(label_path)
        dataset[LABEL_GOLD] = np.asarray(label_list)

    if sampling:
        dataset = custom_sampling(dataset)

    dataset[TID], dataset[SEQ_LEN] = to_nn_input(dataset[TID], max_seq_len=max_seq_len)

    if with_label:
        output_dim = max(dataset[LABEL_GOLD]) + 1
        return dataset, output_dim
    else:
        return dataset


def custom_sampling(dataset):
    raise NotImplementedError

    n_dim = dataset[LABEL_GOLD].max() + 1
    label_idx = [list() for _ in range(n_dim)]
    for i, label in enumerate(dataset[LABEL_GOLD]):
        label_idx[label].append(i)
    dataset[LABEL_GOLD] = dataset[LABEL_GOLD].tolist()

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

    dataset[LABEL_GOLD] = np.asarray(dataset[LABEL_GOLD])
    return dataset


feed_key = {
    TRAIN: [LABEL_GOLD, TID, SEQ_LEN, ],
    TEST: [TID, SEQ_LEN, ],
}

fetch_key = {
    TRAIN: [OPTIMIZER, LOSS, LABEL_PREDICT],
    TEST: [LABEL_PREDICT, PROB_PREDICT, HIDDEN_FEAT]
}


@commandr.command
def train(text_version='ek', label_version=None, config_path='c83.yaml'):
    """
    python -m algo.main93_v2 train
    python3 -m algo.main93_v2 train -c config_ntua93.yaml

    :param text_version: string
    :param label_version: string
    :param config_path: string
    :return:
    """
    pos_label = 1 if label_version == 'A' else None

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
    datasets = dict()
    datasets[TRAIN], output_dim = load_dataset(
        mode=TRAIN, vocab_id_mapping=vocab_id_mapping,
        max_seq_len=nn_config.seq_len, sampling=train_config.train_sampling,
        label_version=label_version
    )
    datasets[TEST], _ = load_dataset(
        mode=TEST, vocab_id_mapping=vocab_id_mapping, max_seq_len=nn_config.seq_len,
        label_version=label_version
    )

    # 初始化数据集的检索
    index_iterators = {
        TRAIN: IndexIterator.from_dataset(datasets[TRAIN]),
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

    best_res = {mode: None for mode in [TRAIN, VALID]}
    no_update_count = {mode: 0 for mode in [TRAIN, VALID]}
    max_no_update_count = 10

    eval_history = {TRAIN: list(), VALID: list(), TEST: list()}

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
            res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)
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
                res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)
                eval_history[VALID].append(res)
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
            res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)
            eval_history[TEST].append(res)
            print('TEST')
            print_evaluation(res)

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
            if mode == TRAIN and train_config.train_sampling:
                dataset, _ = load_dataset(
                    mode=TRAIN, vocab_id_mapping=vocab_id_mapping,
                    max_seq_len=nn_config.seq_len, sampling=False, label_version=label_version
                )
            else:
                dataset = datasets[mode]
            index_iterator = SimpleIndexIterator.from_dataset(dataset)
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
                if LABEL_GOLD in dataset:
                    labels_gold += dataset[LABEL_GOLD][batch_index].tolist()

            prob_predict = prob_predict[:n_sample]
            labels_predict = labels_predict[:n_sample]
            labels_gold = labels_gold[:n_sample]
            hidden_feats = hidden_feats[:n_sample]

            if mode == TEST:
                res = basic_evaluate(gold=labels_gold, pred=labels_predict, pos_label=pos_label)
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
        for col in res[CONFUSION_MATRIX]:
            print(','.join(map(str, col)))

        json.dump(res, open(data_config.output_path(output_key, mode, EVALUATION), 'w'))
        print()

    test_score_list = map(lambda _item: _item['f1'], eval_history[TEST])
    print('best test f1 reached: {}'.format(max(test_score_list)))

    print('OUTPUT_KEY: {}'.format(output_key))


@commandr.command('test')
def live_test(output_key):
    config_path = data_config.output_path(output_key, ALL, CONFIG)
    config_data = yaml.load(open(config_path))
    nn_config = NNConfig(config_data)
    vocab_id_mapping = json.load(open(data_config.output_path(output_key, ALL, VOCAB_ID_MAPPING), 'r'))

    with tf.Session() as sess:
        prefix_checkpoint = tf.train.latest_checkpoint(data_config.model_path(key=output_key))
        saver = tf.train.import_meta_graph('{}.meta'.format(prefix_checkpoint))
        saver.restore(sess, prefix_checkpoint)

        nn = BaseNNModel(config=None)
        nn.set_graph(tf.get_default_graph())

        fetches = {_key: nn.var(_key) for _key in [LABEL_PREDICT, PROB_PREDICT]}
        while True:
            res = input('input: ')
            if res == 'quit':
                break

            turns = res.strip().split('|')
            if len(turns) != 3:
                print('invalid turns')
                continue

            tokens_list = list()
            for turn in turns:
                tokens = re.sub('\s+', ' ', turn.strip()).split(' ')
                tokens_list.append(tokens)

            placeholder = [[]] * (nn_config.batch_size - 1)
            tid_list_0 = tokenized_to_tid_list([tokens_list[0], ] + placeholder, vocab_id_mapping)
            tid_list_1 = tokenized_to_tid_list([tokens_list[1], ] + placeholder, vocab_id_mapping)
            tid_list_2 = tokenized_to_tid_list([tokens_list[2], ] + placeholder, vocab_id_mapping)

            tid_0 = np.asarray(zero_pad_seq_list(tid_list_0, nn_config.seq_len))
            tid_1 = np.asarray(zero_pad_seq_list(tid_list_1, nn_config.seq_len))
            tid_2 = np.asarray(zero_pad_seq_list(tid_list_2, nn_config.seq_len))

            feed_dict = {
                nn.var(TID_0): tid_0,
                nn.var(TID_1): tid_1,
                nn.var(TID_2): tid_2,
                nn.var(TEST_MODE): 1
            }
            res = sess.run(fetches=fetches, feed_dict=feed_dict)
            label = res[LABEL_PREDICT][0]
            prob = res[PROB_PREDICT][0]
            print('label: {}'.format(label))
            print('prob: {}'.format(prob))


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


@commandr.command('clear')
def clear_output(output_key):
    shutil.rmtree(data_config.output_folder(output_key))
    shutil.rmtree(data_config.model_folder(output_key))


if __name__ == '__main__':
    commandr.Run()
