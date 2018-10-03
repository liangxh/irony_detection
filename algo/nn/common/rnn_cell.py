# -*- coding; utf-8 -*-
import tensorflow as tf


def _dropout(cell, keep_prob):
    return tf.nn.rnn_cell.DropoutWrapper(
            cell, output_keep_prob=keep_prob)


def build_lstm(dim, dropout_keep_prob=None):
    cell = tf.nn.rnn_cell.BasicLSTMCell(
                dim, forget_bias=0.1, state_is_tuple=True)
    if dropout_keep_prob is not None:
        cell = _dropout(cell, dropout_keep_prob)
    return cell


def build_gru(dim, dropout_keep_prob=None):
    cell = tf.nn.rnn_cell.GRUCell(dim)
    if dropout_keep_prob is not None:
        cell = _dropout(cell, dropout_keep_prob)
    return cell


def build_basic(dim, dropout_keep_prob=None):
    cell = tf.nn.rnn_cell.BasicRNNCell(dim)
    if dropout_keep_prob is not None:
        cell = _dropout(cell, dropout_keep_prob)
    return cell
