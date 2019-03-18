# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:33:08 2018

@author: ashima.garg
"""

import tensorflow as tf

class Layer():

    def __init__(self, shape, mean, stddev):
        self.weights = tf.Variable(tf.random_normal(shape=shape, mean=mean, stddev=stddev))
        self.biases = tf.Variable(tf.zeros(shape=[shape[-1]]))

    def feed_forward(self, input_data, stride=None):
        raise NotImplementedError


class Convolution_Layer(Layer):

    def __init__(self, shape, mean, stddev):
        super(Convolution_Layer, self).__init__(shape, mean, stddev)

    def feed_forward(self, input_data, stride):
        conv = tf.nn.conv2d(input_data, self.weights, stride, padding="VALID")
        output_data = tf.nn.relu(tf.nn.bias_add(conv, self.biases))
        return output_data


class Output_Layer(Layer):

    def __init__(self, shape, mean, stddev):
        super(Output_Layer, self).__init__(shape, mean, stddev)

    def feed_forward(self, input_data, stride):
        output_data = tf.nn.bias_add(tf.nn.conv2d(input_data, self.weights, stride, padding="VALID"), self.biases)
        return output_data
