# -*- coding: utf-8 -*-
import tensorflow as tf


ph_input = tf.placeholder(tf.float32, [None, 4, ])
output = tf.cast(tf.argmax(ph_input, 1), tf.int32)


x = [-0.07314727, 0.1166857, 0.0513044, -0.09929467]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(fetches=[output], feed_dict={ph_input: [x, ]})
    print res[0]
