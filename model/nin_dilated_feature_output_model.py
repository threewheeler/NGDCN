# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 11:12 PM
# @Author  : morningstarwang
# @FileName: nin_model.py
# @Blog: wangchenxing.com
import tensorflow as tf


class NINBlock(tf.keras.Model):

    def __init__(self, args):
        super(NINBlock, self).__init__()
        self.args = args
        f1, f2, f3, f4, f5, f6, k1, kr, p1, p2, p3, s1, s2, s3, s4, s5, s_s1, s_s2, s_s3 = self.read_args()
        self.output_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(batch_input_shape=(
                None, self.args.dataset_args['image_size'], self.args.dataset_args['image_size'], 3), filters=f1,
                kernel_size=k1, strides=s1, padding='valid',
                kernel_regularizer=tf.keras.regularizers.l2(kr), ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(f4, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(f5, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=p1, strides=s_s1, padding='same'),
            tf.keras.layers.Conv2D(f1, (k1, k1), strides=s2, padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(kr), ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(f2, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(f3, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=p2, strides=s_s2, padding='same'),
            tf.keras.layers.Conv2D(f1, (k1, k1), strides=s3, padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(kr), ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(f2, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(f3, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=p3, strides=s_s3, padding='same'),
            tf.keras.layers.Conv2D(f1, (k1, k1), strides=s4, padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(kr), ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(f2, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(f3, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(f1, (k1, k1), strides=s5, padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(kr), ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(f3, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(f6, (1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
        ])

    def read_args(self):
        f1 = self.args.model_args['hyper_parameters']['f'][0]
        f2 = self.args.model_args['hyper_parameters']['f'][1]
        f3 = self.args.model_args['hyper_parameters']['f'][2]
        f4 = self.args.model_args['hyper_parameters']['f'][3]
        f5 = self.args.model_args['hyper_parameters']['f'][4]
        f6 = self.args.model_args['hyper_parameters']['f'][5]
        k1 = self.args.model_args['hyper_parameters']['k'][0]
        kr = self.args.model_args['hyper_parameters']['kr']
        p1 = tuple(self.args.model_args['hyper_parameters']['p'][0])
        p2 = tuple(self.args.model_args['hyper_parameters']['p'][1])
        p3 = tuple(self.args.model_args['hyper_parameters']['p'][2])
        s_s1 = tuple(self.args.model_args['hyper_parameters']['s_s'][0])
        s_s2 = tuple(self.args.model_args['hyper_parameters']['s_s'][1])
        s_s3 = tuple(self.args.model_args['hyper_parameters']['s_s'][2])
        s1 = self.args.model_args['hyper_parameters']['s'][0]
        s2 = self.args.model_args['hyper_parameters']['s'][1]
        s3 = self.args.model_args['hyper_parameters']['s'][2]
        s4 = self.args.model_args['hyper_parameters']['s'][3]
        s5 = self.args.model_args['hyper_parameters']['s'][4]
        return f1, f2, f3, f4, f5, f6, k1, kr, p1, p2, p3, s1, s2, s3, s4, s5, s_s1, s_s2, s_s3

    def call(self, inputs, **kwargs):
        return self.output_layers(inputs)


class DilatedBlock(tf.keras.Model):

    def __init__(self, args):
        super(DilatedBlock, self).__init__()
        self.args = args
        f1, f2, f3, f4, f5, f6, k1, k2, kr, p1, p2, p3, p4, s1, s2, s3, s4, s5, s_s1, s_s2, s_s3, dilated = self.read_args()
        self.output_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(batch_input_shape=(
                None, self.args.dataset_args['image_size'], self.args.dataset_args['image_size'], 3), filters=f1,
                kernel_size=k1, strides=1, padding='valid',
                kernel_regularizer=tf.keras.regularizers.l2(kr), dilation_rate=dilated),
            tf.keras.layers.Conv2D(f2, (k1, k1), strides=1, padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(kr), dilation_rate=dilated),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=p1, padding='same'),
            #
            tf.keras.layers.Conv2D(f3, (k1, k1), strides=1, padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(kr), dilation_rate=dilated),
            tf.keras.layers.Conv2D(f4, (k1, k1), strides=1, padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(kr), dilation_rate=dilated),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=p2, padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=p2, padding='same'),
            tf.keras.layers.Conv2D(f6, (k1, k2), strides=1, padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(kr)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            # tf.keras.layers.Conv2D(f5, (k1, k1), strides=s3, padding='valid',
            #                        kernel_regularizer=tf.keras.regularizers.l2(kr), dilation_rate=dilated),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.Activation('relu'),
            # tf.keras.layers.MaxPooling2D(pool_size=p3, padding='same'),
            #
            # tf.keras.layers.Conv2D(f6, (k2, k2), strides=s4, padding='same',
            #                        kernel_regularizer=tf.keras.regularizers.l2(kr), dilation_rate=dilated),

            # tf.keras.layers.MaxPooling2D(pool_size=p4, padding='same'),
        ])

    def read_args(self):
        f1 = self.args.model_args['hyper_parameters']['d_f'][0]
        f2 = self.args.model_args['hyper_parameters']['d_f'][1]
        f3 = self.args.model_args['hyper_parameters']['d_f'][2]
        f4 = self.args.model_args['hyper_parameters']['d_f'][3]
        f5 = self.args.model_args['hyper_parameters']['d_f'][4]
        f6 = self.args.model_args['hyper_parameters']['d_f'][5]
        k1 = self.args.model_args['hyper_parameters']['d_k'][0]
        k2 = self.args.model_args['hyper_parameters']['d_k'][1]
        kr = self.args.model_args['hyper_parameters']['d_kr']
        p1 = tuple(self.args.model_args['hyper_parameters']['d_p'][0])
        p2 = tuple(self.args.model_args['hyper_parameters']['d_p'][1])
        p3 = tuple(self.args.model_args['hyper_parameters']['d_p'][2])
        p4 = tuple(self.args.model_args['hyper_parameters']['d_p'][3])
        s_s1 = tuple(self.args.model_args['hyper_parameters']['d_s_s'][0])
        s_s2 = tuple(self.args.model_args['hyper_parameters']['d_s_s'][1])
        s_s3 = tuple(self.args.model_args['hyper_parameters']['d_s_s'][2])
        s1 = self.args.model_args['hyper_parameters']['d_s'][0]
        s2 = self.args.model_args['hyper_parameters']['d_s'][1]
        s3 = self.args.model_args['hyper_parameters']['d_s'][2]
        s4 = self.args.model_args['hyper_parameters']['d_s'][3]
        s5 = self.args.model_args['hyper_parameters']['d_s'][4]
        dilated = self.args.model_args['hyper_parameters']['dilated']
        return f1, f2, f3, f4, f5, f6, k1, k2, kr, p1, p2, p3, p4, s1, s2, s3, s4, s5, s_s1, s_s2, s_s3, int(dilated)

    def call(self, inputs, **kwargs):
        return self.output_layers(inputs)


class CombinedModel(tf.keras.Model):

    def __init__(self, args):
        super(CombinedModel, self).__init__()
        self.args = args
        self.nin = NINBlock(args)
        self.dilated = DilatedBlock(args)
        # self.output_layers = tf.keras.Sequential([
        #     tf.keras.layers.GlobalAveragePooling2D(),
        #     tf.keras.layers.Activation('softmax')
        # ])

    def call(self, inputs, training=None, mask=None):
        nin_output = self.nin(inputs)
        dilated_output = self.dilated(inputs)
        nin_dilated_output = tf.concat([nin_output, dilated_output], axis=1)
        return nin_dilated_output


class NINDilatedFeatureOutputModel:

    def __init__(self, args):
        self.args = args
        self.model = tf.keras.Sequential([
            CombinedModel(args)
        ])
