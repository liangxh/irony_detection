# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from algo.nn.common.common import mask_by_seq_len

ph_input = tf.Variable(np.asarray(
    [
        [[1, 1], [2, 2], [3, 3]],
        [[10, 10], [20, 20], [30, 30]],
        [[100, 100], [200, 200], [300, 300]],
        [[-100, -100], [-200, -200], [-300, -300]],
    ]
), dtype=tf.float32)
seq_len = tf.Variable(np.asarray(
    [1, 2, 3, 1]
))
y = mask_by_seq_len(ph_input, seq_len)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(fetches=[y])
    print res[0]
