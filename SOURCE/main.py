# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:32:40 2018

@author: ashima.garg
"""

import data
import model
import config

if __name__ == "__main__":
    # READ DATA
    data = data.DATA(config.TRAIN_DIR)
    # BUILD MODEL
    model = model.MODEL()
    model.build()
    # TRAIN MODEL
    model.train(data)
    # TEST MODEL
    data = data.DATA(config.TEST_DIR)
    model.test(data)


