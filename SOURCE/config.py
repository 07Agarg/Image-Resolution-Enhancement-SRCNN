# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 20:31:45 2018

@author: ashima.garg
"""

import os

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')

TRAIN_DIR = "train"
TEST_DIR = "test"

# DATA INFORMATION
INPUT_SIZE = 33
BATCH_SIZE = 5
OUTPUT_SIZE = 21

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# TRAINING INFORMATION
NUM_EPOCHS = 200

INITIAL_LEARNING_RATE = 0.00001 

#IMAGE PROCESSING DETAILS
UPSCALE_FACTOR = 3
PAD = 6
STRIDE = 14