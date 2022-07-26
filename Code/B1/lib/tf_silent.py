# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 09:47:46 2022

@author: zahra
"""

# suppress tensorflow warnings (must be called before importing tensorflow)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)