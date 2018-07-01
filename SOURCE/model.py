# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:34:01 2018

@author: ashima.garg
"""

import tensorflow as tf
import config
import neural_network
import os


class MODEL():

    def __init__(self):
        self.inputs = tf.placeholder(shape=[config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, 1], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[config.BATCH_SIZE, config.IMAGE_SIZE, config.IMAGE_SIZE, 1], dtype=tf.float32)
        self.loss = None
        self.output = None

    def build(self):
        input_data = self.inputs

        conv1 = neural_network.Convolution_Layer(shape= [9, 9, 1, 64], mean = 0.0, stddev = 0.001)
        h = conv1.feed_forward(input_data=input_data, stride=[1, 1, 1, 1])

        conv2 = neural_network.Convolution_Layer(shape = [1, 1, 64, 32], mean = 0.0, stddev = 0.001)
        h = conv2.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        outer_layer = neural_network.Outer_layer(shape = [5, 5, 32, 1], mean = 0.0, stddev = 0.001)
        logits = outer_layer.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        logits_norm = tf.image.convert_image_dtype(logits, tf.float32)/255.
        self.output = logits_norm
        #self.output = tf.image.resize_images(logits_norm, [224, 224], method=tf.image.ResizeMethod.BICUBIC)
        self.loss = tf.reduce_mean(tf.squared_difference(self.labels, self.output))

    def train(self, data):
       # global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(0.01, global_step, data.size, 0.95, staircase=True)
        optimizer = tf.train.AdamOptimizer(config.INITIAL_LEARNING_RATE).minimize(self.loss)
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print('All variables Initialized')
            total_batch = int(data.size/config.BATCH_SIZE)
            for epoch in range(config.NUM_EPOCHS):
                avg_cost = 0
                for batch in range(total_batch):
                    batch_X, batch_Y = data.generate_batch()
                    feed_dict = {self.inputs: batch_X, self.labels: batch_Y}
                    _, loss_val = session.run([optimizer, self.loss], feed_dict=feed_dict)
                    print("batch:", batch, " loss: ", loss_val)
                    avg_cost += loss_val / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

            save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            print("Model saved in path: %s" % save_path)

    def test(self, data):
        with tf.Session() as session:
            saver = tf.train.Saver()
            saver.restore(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            avg_cost = 0
            total_batch = int(data.size/config.BATCH_SIZE)
            for batch in range(total_batch):
                batch_X, batch_Y = data.generate_batch()
                feed_dict = {self.inputs: batch_X, self.labels: batch_Y}
                pred_Y, loss = session.run([self.output, self.loss], feed_dict=feed_dict)
                pred_Y = self.deprocess(pred_Y)
                self.reconstruct(batch_X, pred_Y)
                avg_cost += loss/total_batch
            print("cost =", "{:.3f}".format(avg_cost))
        