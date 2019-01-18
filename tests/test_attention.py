# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from algo.nn.common import attention

ph_input = tf.Variable(np.asarray(
    [
        [[1, -1], [2, -2], [3, -3]],
        [[4, -4], [5, -5], [6, -6]],
    ]
), dtype=tf.float32)

ph_context = tf.Variable(np.asarray(
    [
        [1, 2, 3],
        [4, 5, 6],
    ]
), dtype=tf.float32)


fetches = attention.build2(ph_input=ph_input, ph_context=ph_context)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(fetches=fetches)
    for item in res:
        print item
