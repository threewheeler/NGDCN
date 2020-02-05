# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 9:46 PM
# @Author  : morningstarwang
# @FileName: nin_instance.py
# @Blog: wangchenxing.com
import datetime

from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

import dataset
import model
import tensorflow as tf

from decor import exec_feature_output


class GCNInstance:
    def __init__(self, args):
        self.args = args
        self.dataset = None
        self.model = None
        self.optimizer = tf.keras.optimizers.Adam()

        self.epoch_loss_avg = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.epoch_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy', dtype=tf.float32)
        self.validate_accuracy = tf.keras.metrics.Accuracy('test_accuracy', dtype=tf.float32)
        self.validate_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.train_loss_results = []
        self.train_accuracy_results = []
        self.validate_accuracy_results = [0]
        self.load_data()
        self.load_model()
        # self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        # self.manager = tf.train.CheckpointManager(self.ckpt, f'.\\tf_ckpts_{self.args.model_args["model_prefix"]}',
        #                                           max_to_keep=10)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def print_log(self, content):
        print(content)
        with open(f'{self.args.model_args["model_prefix"]}_log.txt', 'a') as f:
            print(content, file=f)

    def load_data(self):
        self.dataset = getattr(dataset, self.args.dataset)(self.args, self.print_log)

    def load_model(self):
        self.model = getattr(model, self.args.model)(self.args).model

    # @tf.function
    def train_step(self, features, adj, labels):
        labels = tf.one_hot(labels, depth=8)
        with tf.GradientTape() as tape:
            logits = self.model([features, adj], training=True)
            loss_value = tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.epoch_loss_avg(loss_value)
        self.epoch_accuracy(labels, logits)

    def train_and_validate(self, epochs, is_test=False):
        # self.ckpt.restore(self.manager.latest_checkpoint)
        # if self.manager.latest_checkpoint:
        #     self.print_log("Restored from {}".format(self.manager.latest_checkpoint))
        # else:
        #     self.print_log("Initializing from scratch.")
        for epoch in range(epochs):
            t = tqdm(total=self.dataset.train_steps_per_epoch, leave=True)
            self.model.training = True
            for (batch, (features, adj, labels)) in enumerate(self.dataset.train_ds):
                self.train_step(features, adj, labels)
                if is_test:
                    return
                # self.ckpt.step.assign_add(1)
                # if int(self.ckpt.step) % 100 == 0:
                #     save_path = self.manager.save()
                t.set_description(
                    f"Epoch {epoch + 1:03d}: Loss: {self.epoch_loss_avg.result():.3f}, Accuracy: {self.epoch_accuracy.result():.3%}")
                t.update(1)
            t.close()
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.epoch_loss_avg.result(), step=epoch)
                tf.summary.scalar('accuracy', self.epoch_accuracy.result(), step=epoch)
            self.train_loss_results.append(self.epoch_loss_avg.result())
            self.train_accuracy_results.append(self.epoch_accuracy.result())
            self.epoch_loss_avg.reset_states()
            self.epoch_accuracy.reset_states()
            self.model.training = False
            for (batch, (features, adj, labels)) in enumerate(self.dataset.validate_ds):
                try:
                    self.validate_step(features, adj, labels)
                except:
                    self.print_log("Find corrupt data, skip this validating step.")
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.validate_loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', self.validate_accuracy.result(), step=epoch)
            if self.validate_accuracy.result() > max(self.validate_accuracy_results):
                self.model.save_weights(
                    f"{self.args.model_args['model_prefix']}_{self.args.instance}_{self.args.dataset}.hdf5")
                self.print_log(
                    f"Validate accuracy updated from "
                    f"{max(self.validate_accuracy_results):.3%} to {self.validate_accuracy.result():.3%}")
            self.print_log(f"Validate accuracy: {self.validate_accuracy.result():.3%}")
            self.validate_accuracy_results.append(self.validate_accuracy.result())
            self.validate_accuracy.reset_states()
            self.validate_loss.reset_states()

    # @tf.function
    def validate_step(self, features, adj, labels):
        logits = self.model([features, adj])
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        self.validate_accuracy(prediction, labels)
        labels = tf.one_hot(labels, depth=8)
        loss = tf.keras.losses.categorical_crossentropy(labels, logits)
        self.validate_loss(loss)

    def test(self):
        weight_file_name = f"{self.args.model_args['model_prefix']}_{self.args.instance}_{self.args.dataset}.hdf5"
        self.print_log(f"Testing on weight file:{weight_file_name}")
        # self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics="accuracy")
        self.model.load_weights(weight_file_name)
        # self.model.save_weights(
        #     f"{self.args.model_args['model_prefix']}_{self.args.instance}_{self.args.dataset}cp.hdf5")
        for (batch, (features, adj, labels)) in enumerate(self.dataset.validate_ds):
            logits = self.model.predict([features, adj])
            prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
            cf = confusion_matrix(labels, prediction)
            cr = classification_report(labels, prediction,
                                       target_names=['Still', 'Walk', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway'])
            self.print_log(cf)
            self.print_log(cr)

    def exec(self):
        if 'train' == self.args.instance_args["mode"]:
            self.train_and_validate(self.args.instance_args["epoch"])
        elif 'test' == self.args.instance_args["mode"]:
            self.train_and_validate(self.args.instance_args["epoch"], is_test=True)
            self.test()
