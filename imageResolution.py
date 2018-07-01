# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:58:17 2018

@author: ashima.garg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import re
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.misc import imsave 

resolved = 'Resolved'
INPUT_SIZE = 33
OUTPUT_SIZE = 21
NUM_EPOCHS = 100
EVAL_FREQUENCY = 1
#FLAGS = tf.app.flags.FLAGS
'''
tf.app.flags.DEFINE_integer('batch_size', 1,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('batch_size_test', 1,"""Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")
'''
batch_size = 1
INITIAL_LEARNING_RATE = 0.0001       # Initial learning rate.
scaleX = 2
scaleY = 2
x = 100
y = 100
Upscale_Factor = 3
stride = 14
pad = 6
count = 0
'''
def data_type():
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32
'''

def display_img(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)

def preprocess_img(img):
    height, width, channels = img.shape
    print("read img shape "+ str(img.shape))
    ycbcrImg = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    #display_img(ycbcrImg, "YCBCRIMG")
    
    ycbcrImg = ycbcrImg.astype(np.float)/255.
    print("shape of ycbcr img " + str(ycbcrImg.shape))
    label = ycbcrImg
    blurImg = cv2.GaussianBlur(ycbcrImg, (5,5), 0)     #kernel size = 5,5 ; check with other also. 
    #display_img(blurImg, "BLUR IMAGE")
    
    subsampleImg = cv2.resize(blurImg, (0,0), fx = 1.0/Upscale_Factor, fy = 1.0/Upscale_Factor, interpolation = cv2.INTER_CUBIC)    
    #display_img(subsampleImg, "SUBSAMPLE IMAGE")
    upscaleImg = cv2.resize(subsampleImg, (0, 0), fx = Upscale_Factor, fy = Upscale_Factor, interpolation = cv2.INTER_CUBIC)
   # display_img(upscaleImg, "UPSCALE IMAGE")
    height, width, channels = upscaleImg.shape
    print("height, " + str(height) + " width " + str(width))
    inputx = upscaleImg
    x_ = []
    y_ = []
    global count 
    for i in range(0, height - INPUT_SIZE, stride):
        for j in range(0, width - INPUT_SIZE, stride):
            inputX = inputx[i:i + INPUT_SIZE, j:j + INPUT_SIZE]  
            labelX = label[i+pad:i + pad + OUTPUT_SIZE, j+pad:j + pad + OUTPUT_SIZE]
           # print("input X shape " +str(inputX.shape))
           # print("labelX shape "+str(labelX.shape))
            inputX1 = inputX[:, :, 0]
            labelX1 = labelX[:, :, 0]
            inputX1 = np.atleast_3d(inputX1)
            labelX1 = np.atleast_3d(labelX1)
           # display_img(inputX1, 'X1 input')
           # display_img(labelX1, 'X1 label')
            '''
            inputX2 = inputX.reshape([INPUT_SIZE, INPUT_SIZE, 1])
            labelX2 = labelX.reshape([OUTPUT_SIZE, OUTPUT_SIZE, 1])
            display_img(inputX2, 'X2 label')
            display_img(labelX2, 'X2 label')
            '''
            x_.append(inputX1) 
            y_.append(labelX1)
            count += 1
    print("inputX shape " + str(inputX.shape))
    print("labelX shape " + str(labelX.shape)) 
    print("Count of total sub-images" + str(count))
    return x_, y_

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h*size[0], w*size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def create(dirname):
    filenames = os.listdir(dirname)
    print("dirname " + str(dirname))
    x_img = []
    y_img = []
    for filename in filenames:
        #grey, color = read_img(dirname+filename)
        #grey_3 = np.atleast_3d(grey)       
        img = cv2.imread(dirname + filename)
     #   cv2.imshow("Input Image", img)
     #   cv2.waitKey(0)
        x, y = preprocess_img(img)
        x_img.extend(x)
        y_img.extend(y)
        '''
        interp, color = read_img(dirname + filename)
        color_l = color[:,:,0]
        color_l3 = np.atleast_3d(color_l)
        #color_cbcr = color[:,:,1:]
        interp_l = interp[:, :, 0]
        interp_l3 = np.atleast_3d(interp_l)
        x_img.append(color_l3)
        y_img.append(interp_l3)
        '''
    x = np.asarray(x_img)
    y = np.asarray(y_img)
    return x, y

def create_network(image):
    with tf.variable_scope('level1') as scope:
        weight = tf.Variable(tf.random_normal(shape = [9, 9, 1, 64], mean = 0.0, stddev = 0.001))
        bias = tf.Variable(tf.zeros(shape = [64]))
        conv = tf.nn.conv2d(image, weight, [1, 1, 1, 1], padding = "VALID")
        level1 = tf.nn.relu(tf.nn.bias_add(conv, bias))
    
    with tf.variable_scope('level2') as scope:
        weight = tf.Variable(tf.random_normal(shape = [1, 1, 64, 32], mean = 0.0, stddev = 0.001))
        bias = tf.Variable(tf.zeros(shape = [32]))
        conv = tf.nn.conv2d(level1, weight, [1, 1, 1, 1],padding = "VALID")
        level2 = tf.nn.relu(tf.nn.bias_add(conv, bias))
    
    with tf.variable_scope('level3') as scope:
        weight = tf.Variable(tf.random_normal(shape = [5, 5, 32, 1], mean = 0.0, stddev = 0.001))
        bias = tf.Variable(tf.zeros(shape = [1]))
        level3 = tf.nn.conv2d(level2, weight, [1, 1, 1, 1],padding = "VALID")
        #level3 = tf.nn.relu(tf.nn.bias_add(conv, bias))
    
    return level3

def trainNetwork():
    path = os.path.dirname(os.path.realpath(__file__))
    path = "C:\\Users\\brije\\Desktop\\Image resolution Enhancement"
    file = "\\train\\"
    dirname = path + file
    x_img_train, y_img_train = create(dirname)
    x_img_train_shape = x_img_train.shape
    train_size = x_img_train_shape[0]
    
    file = "\\test\\"
    dirname = path+file
    x_img_test, y_img_test = create(dirname)
    
    #x_temp = np.reshape(x_img_train , (x_img_train_shape[0], x_img_train_shape[1], x_img_train_shape[2], x_img_train_shape[3]))
    #x_img_train : grayscale images 
    X_input_train = tf.placeholder(dtype = tf.float32, shape = (None, INPUT_SIZE, INPUT_SIZE, 1), name = "input_tensor")
    Y_output_train = tf.placeholder(dtype = tf.float32, shape = (None, OUTPUT_SIZE, OUTPUT_SIZE, 1), name = "output_tensor")
    
    logits = create_network(X_input_train)
    
    #mean squared error
    loss = tf.reduce_mean(tf.square((Y_output_train) - (logits)))
    
    optimizer = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(loss)
    
    with tf.Session() as sess:
        print("session created")
        init = tf.global_variables_initializer()
        sess.run(init)
        start_time = time.time()    
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_checkpoint")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
            
        counter = 0            
        
        for step in xrange(int(NUM_EPOCHS * train_size) // batch_size):
          offset = (step * batch_size) % (train_size - batch_size)
          
          batch_data = x_img_train[offset:(offset + batch_size)]
          batch_labels = y_img_train[offset:(offset + batch_size)]
          counter += 1
          feed_dict = {X_input_train: batch_data, Y_output_train: batch_labels}
          _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
          if counter % 10 == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Step %d (epoch %.2f), %.1f ms' %(step, float(step) * batch_size / train_size, 1000 * elapsed_time / EVAL_FREQUENCY))
            print('Minibatch loss: %.3f' % (l))
          if counter % 500 == 0:
              saver.save(sess, 'saved_checkpoint/'+resolved, global_step = counter)    
        
        feed_dict = {X_input_train: x_img_test, Y_output_train: y_img_test}
        result = create_network.eval(feed_dict= feed_dict)
        result = merge(result, [nx, ny])
        result = result.squeeze()
        path = "outputs\\"
        scipy.misc.imsave(path, result)
    print("training completed")
    

if __name__ == '__main__':
    #sess = tf.InteractiveSession()
    trainNetwork()
    print("Training Completed")