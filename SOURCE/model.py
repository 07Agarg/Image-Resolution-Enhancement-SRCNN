# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:34:01 2018

@author: ashima.garg
"""

import tensorflow as tf
import config
import neural_network
import numpy as np
import os
import utils


class MODEL():

    def __init__(self):
        self.inputs = tf.placeholder(shape=[None, config.INPUT_SIZE, config.INPUT_SIZE, 3], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None, config.OUTPUT_SIZE, config.OUTPUT_SIZE, 3], dtype=tf.float32)
        self.logits = None
        self.output = None
        self.loss = None

    def build(self):
        input_data = self.inputs

        conv1 = neural_network.Convolution_Layer(shape= [9, 9, 3, 64], mean = 0.0, stddev = 0.001, value = 0.1)
        h = conv1.feed_forward(input_data=input_data, stride=[1, 1, 1, 1])

        conv2 = neural_network.Convolution_Layer(shape = [1, 1, 64, 32], mean = 0.0, stddev = 0.001, value = 0.1)
        h = conv2.feed_forward(input_data=h, stride=[1, 1, 1, 1])

        outer_layer = neural_network.Output_Layer(shape = [5, 5, 32, 3], mean = 0.0, stddev = 0.001, value = 0.1)
        self.logits = outer_layer.feed_forward(input_data=h, stride=[1, 1, 1, 1])
        self.output = tf.image.resize_images(self.logits, [config.OUTPUT_SIZE, config.OUTPUT_SIZE], method=tf.image.ResizeMethod.BILINEAR)
        self.loss = tf.reduce_mean(tf.squared_difference(self.labels, self.output))

    def train(self, data):
        optimizer = tf.train.AdamOptimizer(config.INITIAL_LEARNING_RATE).minimize(self.loss)
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            print('All variables Initialized')
            for epoch in range(config.NUM_EPOCHS):
                avg_cost = 0
                total_batch = 0
                for batch in range(int(data.size*(config.NUM_SUBIMG/config.BATCH_SIZE))):
                    batch_X, batch_Y = data.generate_batch()
                    feed_dict = {self.inputs: batch_X, self.labels: batch_Y}
                    _, loss_val = session.run([optimizer, self.loss], feed_dict=feed_dict)
#                    print("batch:", batch, " loss: ", loss_val)
                    total_batch += 5
                    avg_cost += loss_val
                avg_cost = avg_cost / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

            save_path = saver.save(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            print("Model saved in path: %s" % save_path)

    def test(self, data):
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            avg_cost = 0
            total_batch = int(data.size/config.BATCH_SIZE)
            for batch in range(total_batch):
                batch_X, batch_Y = data.generate_batch()
                feed_dict = {self.inputs: batch_X, self.labels: batch_Y}
                pred_Y, loss = session.run([self.output, self.loss], feed_dict=feed_dict)
                avg_cost += loss/total_batch
            print("cost =", "{:.3f}".format(avg_cost))

    def sr_generate(self, data):
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, os.path.join(config.MODEL_DIR, "model" + str(config.BATCH_SIZE) + "_" + str(config.NUM_EPOCHS) + ".ckpt"))
            for file in data.filelist[:1]:
                data.process_img(file)
                batch = np.asarray(data.batch)
                feed_dict = {self.inputs: batch}
                patches = session.run(self.output, feed_dict=feed_dict)
                image = utils.stitch(patches)
                utils.reconstruct(image, file)
