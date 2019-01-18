# -*- coding: utf-8 -*-
import tensorflow as tf


def add_gaussian_noise_layer(input_layer, stddev, test_mode):
    if stddev == 0.:
        return input_layer

    noise = tf.random_normal(shape=input_layer.get_shape(), mean=0., stddev=stddev, dtype=tf.float32)
    return tf.cond(
        tf.equal(test_mode, tf.constant(1, dtype=tf.int8)),
        lambda: input_layer,
        lambda: input_layer + noise
    )


def build_dropout_keep_prob(keep_prob, test_mode):
    return tf.cond(
        tf.equal(test_mode, tf.constant(1, dtype=tf.int8)),
        lambda: tf.constant(1., dtype=tf.float32),
        lambda: tf.constant(keep_prob, dtype=tf.float32)
    )


def mask_by_seq_len(ph_input, seq_len):
    max_len = ph_input.shape[-2]
    mask = tf.sequence_mask(seq_len, maxlen=max_len, dtype=tf.float32)
    mask = tf.expand_dims(mask, -1)
    return tf.multiply(ph_input, mask)
