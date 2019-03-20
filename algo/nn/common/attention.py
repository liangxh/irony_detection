# -*- coding: utf-8 -*-
import tensorflow as tf


def build(inputs, attention_size):
    inputs_shape = inputs.shape
    seq_length = inputs_shape[1].value
    hidden_size = inputs_shape[2].value

    w = tf.Variable(tf.truncated_normal([hidden_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.truncated_normal([1, attention_size], stddev=0.1))
    u = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), w) + b)  # [batch_size*seq_len, attention_size]
    vu = tf.matmul(v, u)  # [batch_size * seq_len, 1]
    exps = tf.reshape(tf.exp(vu), [-1, seq_length])  # [batch_size, seq_len]
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])  # /[batch_size] -> [batch_size, seq_len]
    output = tf.reduce_sum(
        inputs * tf.reshape(alphas, [-1, seq_length, 1]), 1)
    return output, alphas


def build2(ph_input, ph_context):
    """
    :param ph_input:    [batch_size, max_seq_len, dim_input]
    :param ph_context:  [batch_size, dim_context]
    :return:
    """
    batch_size, max_seq_len, dim_input = ph_input.shape.as_list()
    dim_context = ph_context.shape[-1].value

    _ph_input = tf.reshape(ph_input, [-1, dim_input])
    _ph_input = tf.transpose(_ph_input, [0, 1])

    w = tf.Variable(tf.truncated_normal([dim_input, dim_context], stddev=0.1))
    w_context = tf.matmul(a=w, b=ph_context, transpose_b=True)  # [dim_input, batch_size]
    w_context = tf.transpose(w_context)

    w_context = tf.tile(w_context, [1, max_seq_len])
    w_context = tf.reshape(w_context, [-1, dim_input])
    scores = tf.reduce_sum(tf.multiply(_ph_input, w_context), axis=1)

    exps = tf.reshape(tf.exp(scores), [-1, max_seq_len])  # [batch_size, seq_len]
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])  # /[batch_size] -> [batch_size, seq_len]
    output = tf.reduce_sum(
        ph_input * tf.reshape(alphas, [-1, max_seq_len, 1]), 1)
    return output, alphas
