# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:33:23 2018

@author: ashima.garg
"""

import numpy as np
import cv2
import os
import config

def modcrop(image, scale=3):
    if image.shape[2] == 3:
        size = image.shape
        size -= np.mod(size, scale)
        image = image[0:size[0], 0:size[1]]
        return image

def im2double(im):
    info = np.iinfo(im.dtype) 
    return im.astype(np.float) / info.max

class DATA():

    def __init__(self, dirname):
        self.dir_path = os.path.join(config.DATA_DIR, dirname)
        self.filelist = os.listdir(self.dir_path)
        self.size = len(self.filelist)
        self.file_index = 0
        self.data_index = 0
        self.batch = None
        self.batch_label = None

    def read_img(self, filename):
        img = cv2.imread(filename, 3)
        height, width, channels = img.shape
        ycr_cbimg = cv2.cvtColor(cv2.resize(img, (config.READ_SIZE, config.READ_SIZE)), cv2.COLOR_BGR2YCR_CB)
        return ycr_cbimg

    def preprocess_img(self, img):
        img = im2double(img)
        labels = modcrop(img, config.SCALE)
        bicubic_img = cv2.resize(img, None, fx=1.0/config.SCALE, fy=1.0/config.SCALE, interpolation=cv2.INTER_CUBIC)
        inputs = cv2.resize(bicubic_img, None, fx=config.SCALE/1.0, fy=config.SCALE/1.0, interpolation=cv2.INTER_CUBIC)
        inputs = inputs[:, :, 0]
        labels = labels[:, :, 0]
        return inputs, labels

    def process_img(self, file):
        self.batch = []
        self.batch_label = []
        img = self.read_img(os.path.join(config.DATA_DIR, self.dir_path, file))
        inputs, labels = self.preprocess_img(img)
        nx = ny = 0
        for i in range(0, config.READ_SIZE - config.INPUT_SIZE, config.STRIDE):
            nx += 1
            ny = 0
            for j in range(0, config.READ_SIZE - config.INPUT_SIZE, config.STRIDE):
                ny += 1
                sub_input = inputs[i:(i+config.INPUT_SIZE), j:(j+config.INPUT_SIZE)]
                sub_label = labels[(i+config.PAD):(i+config.PAD+config.OUTPUT_SIZE), (j+config.PAD):(j+config.PAD+config.OUTPUT_SIZE)]
                sub_input = np.reshape(sub_input, (config.INPUT_SIZE, config.INPUT_SIZE, config.DIM))
                sub_label = np.reshape(sub_label, (config.OUTPUT_SIZE, config.OUTPUT_SIZE, config.DIM))
                self.batch.append(sub_input)
                self.batch_label.append(sub_label)

    def generate_batch( self):
        if self.data_index >= (config.NUM_SUBIMG-1) or self.batch is None:
            self.data_index = 0
            self.process_img(self.filelist[self.file_index])
            self.file_index = (self.file_index + 1) % self.size
        batch = np.asarray(self.batch[self.data_index:(self.data_index+config.BATCH_SIZE)])
        label = np.asarray(self.batch_label[self.data_index:(self.data_index+config.BATCH_SIZE)])
        self.data_index = self.data_index + config.BATCH_SIZE
        return batch, label


def DataTests():
    data = DATA(config.TEST_DIR)
    for i in range(int(2*(config.NUM_SUBIMG/config.BATCH_SIZE))):
        batch, label = data.generate_batch()
        print(batch.shape)
        print(i)

# DataTests()