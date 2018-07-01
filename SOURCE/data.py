# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:33:23 2018

@author: ashima.garg
"""


import pandas as pd
import numpy as np
import cv2
import os
import config


class DATA():

    def __init__(self, dirname):
        self.dir_path = os.path.join(config.DATA_DIR, dirname)
        self.filelist = os.listdir(self.dir_path)
        self.batch_size = config.BATCH_SIZE
        self.size = len(self.filelist)
        self.data_index = 0

    def preprocess_img(img):
        height, width, channels = img.shape
        ycbcrImg = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        ycbcrImg = ycbcrImg.astype(np.float)/255.
        label = ycbcrImg
        blurImg = cv2.GaussianBlur(ycbcrImg, (5,5), 0)     #kernel size = 5,5 ; check with other also. 
        subsampleImg = cv2.resize(blurImg, (0,0), fx = 1.0/config.UPSCALE_FACTOR, fy = 1.0/config.UPSCALE_FACTOR, interpolation = cv2.INTER_CUBIC)    
        upscaleImg = cv2.resize(subsampleImg, (0, 0), fx = config.UPSCALE_FACTOR, fy = config.UPSCALE_FACTOR, interpolation = cv2.INTER_CUBIC)
        height, width, channels = upscaleImg.shape
        inputx = upscaleImg
        x_ = []
        y_ = []
        #global count 
        for i in range(0, height - config.INPUT_SIZE, config.STRIDE):
            for j in range(0, width - config.INPUT_SIZE, config.STRIDE):
                inputX = inputx[i:i + config.INPUT_SIZE, j:j + config.INPUT_SIZE]  
                labelX = label[i + config.PAD:i + config.PAD + config.OUTPUT_SIZE, j + config.PAD:j + config.PAD + config.OUTPUT_SIZE]
                inputX1 = inputX[:, :, 0]
                labelX1 = labelX[:, :, 0]
                inputX1 = np.reshape(inputX1, (config.INPUT_SIZE, config.INPUT_SIZE, 1))
                labelX1 = np.reshape(labelX1, (config.OUTPUT_SIZE, config.OUTPUT_SIZE, 1))
                #inputX1 = np.atleast_3d(inputX1)
                #labelX1 = np.atleast_3d(labelX1)
                x_.append(inputX1) 
                y_.append(labelX1)
                #count += 1
        return x_, y_

    def read_img(self, filename):
        img = cv2.imread(filename, 3)
        height, width, channels = img.shape
        return self.preprocess_img(img)
        
    def generate_batch(self):
        batch = []
        labels = []
        for i in range(self.batch_size):
            filename = os.path.join(config.DATA_DIR, self.dir_path, self.filelist[self.data_index])
            print(filename)
            x_batch, y_batch = self.read_img(filename)
            batch.extend(x_batch)
            labels.extend(y_batch)
            self.data_index = (self.data_index + 1) % self.size
        batch = np.asarray(batch)
        labels = np.asarray(labels)
        return batch, labels