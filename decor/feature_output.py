# -*- coding: utf-8 -*-
# @Time    : 2019/12/31 14:16
# @Author  : morningstarwang
# @Blog    : wangchenxing.com
# @File    : feature_output.py
import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib
from sklearn.model_selection import train_test_split


def exec_feature_output(entrance: object) -> object:
    """
    Add feature output to exec function
    :param entrance: exec()
    :return: exec()
    """
    def decorate(*args):
        self = args[0]
        entrance(self)
        if 'feature_output' == self.args.instance_args["mode"]:
            feature_output(self)

    return decorate


def exec_feature_output_dilated(entrance):
    """
    Add feature output dilated to exec function
    :param entrance: exec()
    :return: exec()
    """
    def decorate(*args):
        self = args[0]
        entrance(self)
        if 'feature_output' == self.args.instance_args["mode"]:
            feature_output_dilated(self)

    return decorate


def prepare_data_feature_output(entrance):
    def decorate(*args):
        self = args[0]
        entrance(self)
        if 'feature_output' == self.args.instance_args["mode"]:
            train_path = self.args.dataset_args['train_path']
            train_root = pathlib.Path(train_path)
            self.train_all_image_paths_for_test = list(train_root.glob('*/*'))
            self.train_all_image_paths_for_test = [str(path) for path in self.train_all_image_paths_for_test]
            split_paths = train_test_split(self.train_all_image_paths_for_test, shuffle=True,
                                           test_size=0.9995,
                                           random_state=self.args.dataset_args['random_seed'])
            self.train_all_image_paths_for_test = split_paths[0]
            train_label_names = sorted(item.name for item in train_root.glob('*/') if item.is_dir())
            train_label_to_index = dict((name, index) for index, name in enumerate(train_label_names))
            train_all_image_labels = [train_label_to_index[pathlib.Path(path).parent.name]
                                      for path in self.train_all_image_paths]
            train_path_ds = tf.data.Dataset.from_tensor_slices(self.train_all_image_paths)
            train_image_ds = train_path_ds.map(self.load_and_preprocess_image)
            train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_all_image_labels, tf.int64))
            train_image_label_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))
            self.train_ds_for_test = train_image_label_ds.cache()
            self.train_ds_for_test = self.train_ds_for_test.shuffle(buffer_size=self.args.dataset_args['batch_size'],
                                                  seed=self.args.dataset_args['random_seed'])
            self.train_ds_for_test = self.train_ds_for_test.batch(len(self.train_all_image_paths_for_test)).prefetch(
                buffer_size=self.args.dataset_args['batch_size'])
            self.train_ds = self.train_ds.batch(self.args.dataset_args['batch_size']).prefetch(
                buffer_size=self.args.dataset_args['batch_size'])
            self.train_steps_per_epoch =  tf.math.ceil(
                len(self.train_all_image_paths) / self.args.dataset_args['batch_size']).numpy()
            self.validate_ds = self.validate_ds.batch(self.args.dataset_args['batch_size']).prefetch(
                buffer_size=self.args.dataset_args['batch_size'])
            self.validate_steps_per_epoch =  tf.math.ceil(
                len(self.validate_all_image_paths) / self.args.dataset_args['batch_size']).numpy()
    return decorate


def feature_output(self):
    weight_file_name = f"{self.args.output_model_name}"
    self.print_log(f"Feature outputting on weight file:{weight_file_name}")
    self.train_and_validate(self.args.instance_args["epoch"], is_test=True)
    self.model.load_weights(weight_file_name)
    self.print_log(f"Feature outputting on training data...")
    self.train_count = 0
    self.validate_count = 0
    for (batch, (images, labels)) in enumerate(self.dataset.train_ds):
        save_feature_output(self, images, labels, 'train',)
    self.print_log(f"Feature outputting on validating data...")
    for (batch, (images, labels)) in enumerate(self.dataset.validate_ds):
        save_feature_output(self, images, labels, 'validate',)


def save_feature_output(self, images, labels, flag):
    features = self.model.predict(images)
    feature_dimension = features.shape[1] * features.shape[2]
    num_of_node = features.shape[3]
    predicted_labels = tf.argmax(tf.nn.softmax(tf.reduce_mean(features, axis=[1, 2]).numpy()).numpy(), axis=1).numpy()
    labels = labels.numpy()
    correct_idx = []
    wrong_idx = []
    for idx in range(len(predicted_labels)):
        if predicted_labels[idx] == labels[idx]:
            correct_idx.append(idx)
        else:
            wrong_idx.append(idx)
    # TODO Only keep half correct samples
    wrong_idx = wrong_idx[: len(wrong_idx) // 4]
    correct_idx.extend(wrong_idx)
    features = features[correct_idx, :, :]
    labels = labels[correct_idx]
    num_of_sample = len(features)
    labels = np.reshape(labels, (num_of_sample, 1))
    features = np.reshape(features, (num_of_sample, feature_dimension * num_of_node))
    data = np.concatenate([features, labels], axis=1)
    if 'train' == flag:
        if self.train_count == 0:
            columns = []
            for i in range(feature_dimension):
                for j in range(num_of_node):
                    columns.append(f'f{i}n{j}')
            columns.append('mode')
            pd_data = pd.DataFrame(data=data, columns=columns)
            pd_data.to_csv(
                f'/usr/lhy/wms/TeamCity/buildAgent/work/e2dd6dc74d5919ab/feature_outputs/{flag}_{self.args.output_model_name}_{self.args.instance}_{self.args.dataset}_{self.train_count}.csv',
                index=False)
            self.train_count += 1
        else:
            pd_data = pd.DataFrame(data=data, columns=None)
            pd_data.to_csv(
                f'/usr/lhy/wms/TeamCity/buildAgent/work/e2dd6dc74d5919ab/feature_outputs/{flag}_{self.args.output_model_name}_{self.args.instance}_{self.args.dataset}_{self.train_count}.csv',
                index=False)
            self.train_count += 1
    elif 'validate' == flag:
        if self.validate_count == 0:
            columns = []
            for i in range(feature_dimension):
                for j in range(num_of_node):
                    columns.append(f'f{i}n{j}')
            columns.append('mode')
            pd_data = pd.DataFrame(data=data, columns=columns)
            pd_data.to_csv(
                f'/usr/lhy/wms/TeamCity/buildAgent/work/e2dd6dc74d5919ab/feature_outputs/{flag}_{self.args.output_model_name}_{self.args.instance}_{self.args.dataset}_{self.validate_count}.csv',
                index=False)
            self.validate_count += 1
        else:
            pd_data = pd.DataFrame(data=data, columns=None)
            pd_data.to_csv(
                f'/usr/lhy/wms/TeamCity/buildAgent/work/e2dd6dc74d5919ab/feature_outputs/{flag}_{self.args.output_model_name}_{self.args.instance}_{self.args.dataset}_{self.validate_count}.csv',
                index=False)
            self.validate_count += 1


def feature_output_dilated(self):
    weight_file_name = f"{self.args.output_model_name}"
    self.print_log(f"Feature outputting on weight file:{weight_file_name}")
    self.model.load_weights(weight_file_name)
    self.print_log(f"Feature outputting on training data...")
    for (batch, (images, labels)) in enumerate(self.dataset.train_ds):
        save_feature_output_dilated(self, images, labels, 'train')
    self.print_log(f"Feature outputting on validating data...")
    for (batch, (images, labels)) in enumerate(self.dataset.validate_ds):
        save_feature_output_dilated(self, images, labels, 'validate')


def save_feature_output_dilated(self, images, labels, flag):
    features = self.model.predict(images)
    feature_dimension = features.shape[1]
    # predicted_labels = tf.argmax(tf.nn.softmax(tf.reduce_mean(features, axis=[1, 2]).numpy()).numpy(), axis=1).numpy()
    labels = labels.numpy()
    # correct_idx = []
    # for idx in range(len(predicted_labels)):
    #     if predicted_labels[idx] == labels[idx]:
    #         correct_idx.append(idx)
    # features = features[correct_idx, :,:]
    # labels = labels[correct_idx]
    num_of_sample = len(features)
    labels = np.reshape(labels, (num_of_sample, 1))
    features = np.reshape(features, (num_of_sample, feature_dimension))
    data = np.concatenate([features, labels], axis=1)
    columns = []
    for i in range(feature_dimension):
            columns.append(f'f{i}')
    columns.append('mode')
    pd_data = pd.DataFrame(data=data, columns=columns)
    pd_data.to_csv(f'{flag}_{self.args.output_model_name}_{self.args.instance}_{self.args.dataset}.csv', index=False)