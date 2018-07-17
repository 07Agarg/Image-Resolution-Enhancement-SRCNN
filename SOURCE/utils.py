# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:11:56 2018

@author: rahul.ghosh
"""

import cv2
import numpy as np
import config
import os


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY, filelist):
    for i in range(config.BATCH_SIZE):
        result = np.concatenate((batchX[i], predictedY[i]), axis=2)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        save_path = os.path.join(config.OUT_DIR, filelist[i][:-4] + "reconstructed.jpg")
        cv2.imwrite(save_path, result)
        

def stitch(output):
    image = np.ndarray(shape=(config.SUB_IMG*config.OUTPUT_SIZE, config.SUB_IMG*config.OUTPUT_SIZE, 3))
    for row in range(config.SUB_IMG):
        