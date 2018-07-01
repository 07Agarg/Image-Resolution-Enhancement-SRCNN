# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:33:08 2018

@author: ashima.garg
"""

import tensorflow as tf
import config


class Layer():

    def __init__(self, shape, stddev, value):
        self.weights = tf.Variable(tf.random_normal(shape=shape, stddev=stddev))
        self.biases = tf.Variable(tf.constant(value=value, shape=[shape[-1]]))

    def feed_forward(self, input_data, stride=None):
        raise NotImplementedError


class Convolution_Layer(Layer):

    def __init__(self, shape, stddev, value):
        super(Convolution_Layer, self).__init__(shape, stddev, value)

    def feed_forward(self, input_data, stride):
        conv = tf.nn.conv2d(input_data, self.weights, stride, padding="VALID")
        output_data = tf.nn.relu(tf.nn.bias_add(conv, self.biases))
        return output_data

class Output_Layer(Layer):

    def __init__(self, shape, stddev, value):
        super(Output_Layer, self).__init__(shape, stddev, value)

    def feed_forward(self, input_data, stride):
        output_data = tf.nn.conv2d(input_data, self.weights, stride, padding="VALID")
        return output_data

