# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 15:31
# @Author  : morningstarwang
# @Blog    : wangchenxing.com
# @File    : attention_layer.py
from tensorflow.keras import layers
import tensorflow as tf


class AttentionLayer(layers.Layer):
    """
    AttentionLayer
    refer to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, bias=False):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.bias = bias
        self.w = None
        self.b = None

    def build(self, inputs):
        self.w = self.add_weight(name="w", shape=(self.input_dim, self.output_dim), trainable=True, initializer="glorot_uniform")
        self.a = self.add_weight(name="b", shape=(2 * self.output_dim, 1), trainable=True)

    def call(self, inputs):
        features = inputs[0]
        adj = inputs[1]
        h = tf.matmul(features, self.w)
        N = h.shape[0]
        a_input = tf.reshape(tf.concat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1), shape=(N, -1, 2 * self.out_features))
        e = tf.nn.leaky_relu(tf.matmul(a_input, self.a))
        zero_vec = -9e15 * tf.ones_like(e)
        attention = tf.where(adj > 0, e, zero_vec)
        attention = tf.nn.softmax(attention, axis=1)
        h_prime = tf.matmul(attention, h)
        return tf.nn.elu(h_prime)
