# -*- coding: utf-8 -*-
# @Time    : 2019/12/31 4:18 PM
# @Author  : morningstarwang
# @FileName: gcn_model.py
# @Blog: wangchenxing.com
import tensorflow as tf

from layers import GraphConvolutionLayer, AttentionLayer


class GCNBlock(tf.keras.Model):
    def __init__(self, args):
        super(GCNBlock, self).__init__()
        self.args = args
        self.gcns = []
        input_dims = self.args.model_args["hyper_parameters"]["input_dims"]
        output_dims = self.args.model_args["hyper_parameters"]["output_dims"]
        if self.args.instance_args["mode"] == "test":
            dropout = 0
        else:
            dropout = self.args.model_args["hyper_parameters"]["dropout"]
        # self.att = AttentionLayer()
        for idx in range(len(self.args.model_args["hyper_parameters"]["input_dims"])):
            if idx == 0:
                self.gcns.append(
                    GraphConvolutionLayer(input_dims[idx], output_dims[idx], bias=True, input_shape=(196, 8), dropout=dropout))
            else:
                self.gcns.append(GraphConvolutionLayer(input_dims[idx], output_dims[idx], bias=True, dropout=dropout))
        self.flatten_layer = tf.keras.layers.Flatten(name="gcn_flatten")
        self.output_layer = tf.keras.layers.Dense(8, name="gcn_dense")

    def call(self, inputs):
        for idx, gcn in enumerate(self.gcns):
            if idx == 0:
                x = gcn(inputs)
            else:
                x = gcn([x, inputs[1]])
        x = self.flatten_layer(x)
        x = self.output_layer(x)
        return x


class GCNModel():
    def __init__(self, args):
        self.args = args
        self.model = GCNBlock(args)
