# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:32:40 2018

@author: ashima.garg
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import data
import model
import config

if __name__ == "__main__":
    # READ DATA
    train_data = data.DATA(config.TRAIN_DIR)
    # BUILD MODEL
    model = model.MODEL()
    model.build()
    # TRAIN MODEL
    model.train(train_data)
    # TEST MODEL
    test_data = data.DATA(config.TEST_DIR)
    model.test(test_data)
    # SUPERRESOLUTION GENERATE
    model.sr_generate(test_data)
