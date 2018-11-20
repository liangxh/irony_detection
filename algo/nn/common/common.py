# -*- coding: utf-8 -*-
import tensorflow as tf


def add_gaussian_noise_layer(input_layer, stddev, test_mode):
    noise = tf.random_normal(shape=input_layer.get_shape(), mean=0., stddev=stddev, dtype=tf.float32)
    return tf.cond(
        tf.equal(test_mode, tf.constant(1, dtype=tf.int8)),
        lambda: input_layer,
        lambda: input_layer + noise
    )
