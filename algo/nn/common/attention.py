# -*- coding: utf-8 -*-
import tensorflow as tf


def build(inputs, attention_size):
    inputs_shape = inputs.shape
    seq_length = inputs_shape[1].value
    hidden_size = inputs_shape[2].value

    w = tf.Variable(tf.truncated_normal([hidden_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.truncated_normal([1, attention_size], stddev=0.1))
    u = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), w)) + b  # [batch_size*seq_len, attention_size]
    vu = tf.matmul(v, u)  # [batch_size * seq_len, 1]
    exps = tf.reshape(tf.exp(vu), [-1, seq_length])  # [batch_size, seq_len]
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])  # /[batch_size] -> [batch_size, seq_len]
    output = tf.reduce_sum(
        inputs * tf.reshape(alphas, [-1, seq_length, 1]), 1)
    return output, alphas
