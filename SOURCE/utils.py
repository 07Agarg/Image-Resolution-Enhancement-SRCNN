# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:11:56 2018

@author: rahul.ghosh
"""

import cv2
import numpy as np
import config
import os


def reconstruct(image, file):
    result = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR)
    save_path = os.path.join(config.OUT_DIR, file[:-4] + "reconstructed.jpg")
    cv2.imwrite(save_path, result)


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def stitch(patches):
    image = np.ndarray(shape=(config.NUM_SUBIMG*config.OUTPUT_SIZE, config.NUM_SUBIMG*config.OUTPUT_SIZE, 3))
    for index, patch in enumerate(patches):
        j = index % config.NUM_SUBIMG
        i = index // config.NUM_SUBIMG
        image[i*config.OUTPUT_SIZE:(i*config.OUTPUT_SIZE+config.OUTPUT_SIZE), j*config.OUTPUT_SIZE:(j*config.OUTPUT_SIZE+config.OUTPUT_SIZE), :] = patch
    image = deprocess(image)
    return image
