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
    save_path = os.path.join(config.OUT_DIR, file[:-4] + "reconstructed.jpg")
    print("save_path ", save_path)
    cv2.imwrite(save_path, image)

def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)

def stitch(patches):
    image = np.zeros((config.SUB_IMG*config.OUTPUT_SIZE, config.SUB_IMG*config.OUTPUT_SIZE, config.DIM))
    for index, patch in enumerate(patches):
        j = index % config.SUB_IMG
        i = index // config.SUB_IMG
        image[i*config.OUTPUT_SIZE:(i*config.OUTPUT_SIZE+config.OUTPUT_SIZE), j*config.OUTPUT_SIZE:(j*config.OUTPUT_SIZE+config.OUTPUT_SIZE), :] = patch
    image = deprocess(image)
    return image

"""
def display_batch_patch(batch, patch):
    print("batch shape 0: " + str(batch.shape[0]))
    for i in range(batch.shape[0]):
        #print("b1: " + str(batch[i].shape))
        b1 = deprocess(batch[i])
        #print("b1: " + str(b1.shape))
        save_path = os.path.join(config.OUT_DIR, "Batch_" + str(i) + ".jpg")
        cv2.imwrite(save_path, b1)
        p1 = deprocess(patch[i])
        save_path = os.path.join(config.OUT_DIR, "Patch_" + str(i) + ".jpg")
        cv2.imwrite(save_path, p1)
"""    
    