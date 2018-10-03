# -*- coding: utf-8 -*-
import tensorflow as tf


def build(ph_input, filter_num, kernel_size, activation=None):
    if activation is None:
        activation = tf.nn.relu

    filter_shape = [kernel_size, ph_input.shape[-1].value, filter_num]
    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[filter_num]))
    conv = tf.nn.conv1d(ph_input, w, stride=1, padding='VALID')
    h = activation(conv + b)     # [batch_size, seq_len - filter_size + 1, filter_num]

    return h


def max_pooling(ph_input):
    output = tf.nn.max_pool(
                    tf.expand_dims(ph_input, -2),
                    ksize=[1, ph_input.shape[1].value, 1, 1],
                    strides=[1] * 4,
                    padding='VALID'
                )
    output = tf.reshape(output, [-1, output.shape[-1].value])
    return output
