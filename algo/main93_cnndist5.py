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
from algo.lib.dataset import SimpleIndexIterator
from algo.lib.evaluate93 import basic_evaluate
from algo.model.const import *
from algo.model.train_config import TrainConfig
from algo.lib.common import print_evaluation2 as print_evaluation, load_lookup_table2, tokenized_to_tid_list
from algo.model.nn_config import BaseNNConfig
from algo.nn.base import BaseNNModel
from algo.nn.common import dense, cnn
from algo.nn.common.common import add_gaussian_noise_layer, build_dropout_keep_prob
from dataset.common.const import *
from dataset.common.load import *
from dataset.semeval2019_task3_dev.config import config as data_config

TID_ = [TID_0, TID_1, TID_2]
SEQ_LEN_ = [SEQ_LEN_0, SEQ_LEN_1, SEQ_LEN_2]


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
    name = 'm93_cnndist5'
    """
    others的样本人复制一份并随机摘掉一个词
    
    CNN后选择性添加Dense层
    """

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
        tid_1 = tf.placeholder(tf.int32, [self.config.batch_size, self.config.seq_len], name=TID_1)
        tid_2 = tf.placeholder(tf.int32, [self.config.batch_size, self.config.seq_len], name=TID_2)

        embedded_0 = tf.nn.embedding_lookup(lookup_table, tid_0)
        embedded_1 = tf.nn.embedding_lookup(lookup_table, tid_1)
        embedded_2 = tf.nn.embedding_lookup(lookup_table, tid_2)

        if self.config.embedding_noise_type is None:
            pass
        elif self.config.embedding_noise_type == 'gaussian':
            embedded_0 = add_gaussian_noise_layer(embedded_0, stddev=self.config.embedding_noise_stddev, test_mode=test_mode)
            embedded_1 = add_gaussian_noise_layer(embedded_1, stddev=self.config.embedding_noise_stddev, test_mode=test_mode)
            embedded_2 = add_gaussian_noise_layer(embedded_2, stddev=self.config.embedding_noise_stddev, test_mode=test_mode)
        elif self.config.embedding_noise_type == 'dropout':
            emb_dropout_keep_prob = build_dropout_keep_prob(keep_prob=self.config.embedding_dropout_keep_prob, test_mode=test_mode)
            embedded_0 = tf.nn.dropout(embedded_0, emb_dropout_keep_prob)
            embedded_1 = tf.nn.dropout(embedded_1, emb_dropout_keep_prob)
            embedded_2 = tf.nn.dropout(embedded_2, emb_dropout_keep_prob)
        else:
            raise Exception('unknown embedding noise type: {}'.format(self.config.embedding_noise_type))

        with tf.variable_scope("rnn_0") as scope:
            cnn_output = cnn.build2(embedded_0, self.config.filter_num, self.config.kernel_size)
            last_state_0 = cnn.max_pooling(cnn_output)

        with tf.variable_scope("rnn_1") as scope:
            cnn_output = cnn.build2(embedded_1, self.config.filter_num, self.config.kernel_size)
            last_state_1 = cnn.max_pooling(cnn_output)

        with tf.variable_scope("rnn_2") as scope:
            cnn_output = cnn.build2(embedded_2, self.config.filter_num, self.config.kernel_size)
            last_state_2 = cnn.max_pooling(cnn_output)

        dense_input = tf.concat([last_state_0, last_state_1, last_state_2], axis=1, name=HIDDEN_FEAT)
        dense_input = tf.nn.dropout(dense_input, keep_prob=dropout_keep_prob)

        l2_component = None
        for conf in self.config.dense_layers:
            dense_input, w, _ = dense.build(
                dense_input, dim_output=conf['dim'], activation=getattr(tf.nn, conf['activation']))
            if conf.get('l2') > 0:
                comp = conf['l2'] * tf.nn.l2_loss(w)
                if l2_component is None:
                    l2_component = comp
                else:
                    l2_component += comp

        y, w, b = dense.build(dense_input, dim_output=self.config.output_dim, output_name=PROB_PREDICT)

        # 计算loss
        _loss_1 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            logits=y, labels=label_gold, weights=sample_weights))

        _loss_2 = tf.constant(0., dtype=tf.float32)
        if self.config.l2_reg_lambda is not None and self.config.l2_reg_lambda > 0:
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
    # seq_len = seq_to_len_list(tid_list)
    tid_list = np.asarray(zero_pad_seq_list(tid_list, max_seq_len))
    return tid_list  # , seq_len


def load_dataset(vocab_id_mapping, max_seq_len, with_label=True, label_version=None):
    datasets = dict()
    for mode in [TRAIN, TEST]:
        datasets[mode] = dict()
        for i in [0, 1, 2]:
            tid_list = tokenized_to_tid_list(
                load_tokenized_list(data_config.path(mode, TURN, '{}.ek'.format(i))),
                vocab_id_mapping
            )
            datasets[mode][TID_[i]] = tid_list
        if with_label:
            label_path = data_config.path(mode, LABEL, label_version)
            datasets[mode][LABEL_GOLD] = load_label_list(label_path)

    datasets[TRAIN] = custom_sampling(datasets[TRAIN])
    if with_label:
        output_dim = max(datasets[TRAIN][LABEL_GOLD]) + 1
        return datasets, output_dim
    else:
        return datasets


def split_train_valid(dataset, valid_rate, dim=4):
    label_idx = [list() for _ in range(dim)]
    for i, label in enumerate(dataset[LABEL_GOLD]):
        label_idx[label].append(i)

    train = {_key: list() for _key in dataset}
    valid = {_key: list() for _key in dataset}

    for label in range(dim):
        idx_list = label_idx[label]
        n_sample = len(idx_list)
        n_valid = int(n_sample * valid_rate)

        index = list(range(n_sample))
        random.shuffle(index)
        train_index = index[:-n_valid]
        valid_index = index[-n_valid:]

        for i in train_index:
            for _key, _value in dataset.items():
                train[_key].append(_value[idx_list[i]])

        for i in valid_index:
            for _key, _value in dataset.items():
                valid[_key].append(_value[idx_list[i]])

    return train, valid


def dataset_as_input(dataset, max_seq_len):
    new_dataset = dict()
    for i in range(3):
        new_dataset[TID_[i]] = to_nn_input(dataset[TID_[i]], max_seq_len=max_seq_len)
    new_dataset[LABEL_GOLD] = np.asarray(dataset[LABEL_GOLD])
    return new_dataset


def custom_sampling(dataset, dist=None):
    if dist is None:
        dist = [0.88, 0.04, 0.04, 0.04]

    dataset = copy.deepcopy(dataset)
    label_idx = [list() for _ in range(len(dist))]
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
    TRAIN: [TID_0, TID_1, TID_2, LABEL_GOLD],
    TEST: [TID_0, TID_1, TID_2, ]
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
        with_label=True, label_version=label_version
    )
    datasets['all_train'] = datasets[TRAIN]
    datasets[TRAIN], datasets[VALID] = split_train_valid(
        dataset=datasets['all_train'], valid_rate=train_config.valid_rate)
    index_iterators = {
        mode: SimpleIndexIterator.from_dataset(datasets[mode]) for mode in [VALID, TEST, 'all_train']}
    for mode in [VALID, TEST, 'all_train']:
        datasets[mode] = dataset_as_input(datasets[mode], nn_config.seq_len)

    label_weight = {_label: 1. for _label in range(output_dim)}

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

        # 训练开始 ##########################################################################
        for epoch in range(train_config.epoch):
            print('== epoch {} = {} =='.format(epoch, output_key))

            # 利用训练集进行训练
            print('TRAIN')
            dataset = custom_sampling(datasets[TRAIN])
            index_iterator = SimpleIndexIterator.from_dataset(dataset=dataset)
            dataset = dataset_as_input(dataset, nn_config.seq_len)

            n_sample = index_iterator.n_sample()
            labels_predict = list()
            labels_gold = list()

            for batch_index in index_iterator.iterate(batch_size, shuffle=True):
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

                _mode = VALID
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
                eval_history[_mode].append(res)
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
            res = basic_evaluate(gold=labels_gold, pred=labels_predict)
            eval_history[_mode].append(res)
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
            dataset = datasets[mode if mode == TEST else 'all_train']
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
        res = best_res[mode]
        print(mode)
        print_evaluation(res)
        print()

    json.dump(best_res, open(data_config.output_path(output_key, ALL, 'best_eval'), 'w'))

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


if __name__ == '__main__':
    commandr.Run()
