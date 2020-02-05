# -*- coding: utf-8 -*-
# @Time    : 2019/12/31 5:16 PM
# @Author  : morningstarwang
# @FileName: graph_convolution.py
# @Blog: wangchenxing.com
from tensorflow.keras import layers
import tensorflow as tf


class GraphConvolutionLayer(layers.Layer):
    """
    GraphConvolutionLayer
    refer to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, bias=False, **kwargs):
        super(GraphConvolutionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.bias = bias
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_weight(name="gcn_w", shape=(self.input_dim, self.output_dim), trainable=True, initializer="glorot_uniform")
        self.b = self.add_weight(name="gcn_b", shape=(self.output_dim, ), trainable=True)
        self.built = True

    def call(self, inputs):
        features = inputs[0]
        adj = inputs[1]
        # features_zero_like = -9e15 * tf.ones_like(features)
        # features = tf.where(features > 0, features, features_zero_like)

        # adj = tf.ones(adj.shape)
        support = tf.matmul(features, self.w)
        output = tf.matmul(adj, support)
        output = tf.nn.dropout(output, self.dropout)
        # output = self.act(output)
        if self.bias:
            return output + self.b
        else:
            return output
