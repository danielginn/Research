import pandas as pd
import numpy as np
import os
import keras
import matplotlib as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())

base_model = ResNet50(weights='imagenet',include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)  # **** Assuming 2D, with no arguments required
x = Dense(1024, activation='relu', name='fc1')(x)  # **** Assuming relu
delta_xyz = Dense(3, activation='softmax', name='delta_xyz')(x)  # **** Assuming softmax is the correct activation here
delta_rtp = Dense(4, activation='softmax', name='delta_rtp')(x)  # **** Assuming softmax (rho/theta/phi) and quaternians

global_pose_network = Model(inputs=base_model.input, outputs=[delta_xyz, delta_rtp])

global_pose_network.compile(optimizer='Adam',loss='mean_squared_error')
global_pose_network.summary()


