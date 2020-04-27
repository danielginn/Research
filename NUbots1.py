import pandas as pd
import re
import numpy as np
import os
import keras
import scipy
import matplotlib.pyplot as plt
from matplotlib import style
import sys
import time
import random
from matplotlib.pyplot import draw
from keras.layers import Dense, GlobalAveragePooling2D, Activation, concatenate, Reshape, Input, Conv2D, Concatenate, BatchNormalization, Add, Dropout
from keras.initializers import VarianceScaling, Ones
from keras.applications import ResNet50
#from keras.preprocessing import image
from PIL import Image
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.preprocessing.image import load_img
from scipy.spatial.transform import Rotation as R
import ResNet50Modifications as ResNetMods
import LocalisationNetwork
import DatasetInfo

########################################################################################################################
########################################################################################################################
####################################                                               #####################################
####################################  #######  #######  #######  ######   #######  #####################################
####################################  #           #     #     #  #     #     #     #####################################
####################################  #######     #     #######  ######      #     #####################################
####################################        #     #     #     #  #    #      #     #####################################
####################################  #######     #     #     #  #     #     #     #####################################
####################################                                               #####################################
########################################################################################################################
########################################################################################################################


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
print("ResNet50 model loaded...")
#for layer in base_model.layers[:143]: #175 is the final Activation layer: Activation_49, #143 is another one too.
#    layer.trainable = False
#    print(layer.name)


#dropout_rate = 0.2

#base_model = ResNetMods.insert_layer_nonseq(model=base_model, layer_regex='.*activation.*', insert_layer_factory=dropout_layer_factory, position='replace')

#base_model = ResNetMods.insert_feedback_loop(base_model,dropout_rate)

base_model = ResNetMods.additional_final_layers(base_model)

global_pose_network = base_model

global_pose_network.compile(optimizer=Adam(lr=1e-4,epsilon=1e-10),loss='mean_squared_error', metrics=[LocalisationNetwork.xyz_error])
#global_pose_network.summary()

dataset = '7scenes' # Can be: 7scenes, NUbotsField
scene_info = DatasetInfo.GetDatasetInfo(dataset)

######################################################################
###############  Training  ###########################################
######################################################################
print('*****************************')
print('***** STARTING TRAINING *****')
print('*****************************')
style.use('fast')
datagen = ImageDataGenerator(featurewise_center=False)
xyz_avg_error = []
q_avg_error = []
xs = []
file1 = open(".\\Results\\Results.txt", "w")
# Base-line accuracy
test_xyz_error, test_q_error = LocalisationNetwork.Test_epoch(dataset=dataset, scene_info=scene_info, datagen=datagen, model=global_pose_network,
                                          quickTest=False, getPrediction=False)
file1.write("0,,%s,,%s\n" % (test_xyz_error, test_q_error))
file1.close()
xs.append(0)
xyz_avg_error.append(test_xyz_error)
q_avg_error.append(test_q_error)

# Train many epochs
epoch_max = 300
epochs_per_result = 2
result_index = epochs_per_result
for epoch in range(1, epoch_max + 1):
    print('Epoch: ', epoch, '/', epoch_max, sep='')
    global_pose_network, train_xyz_error, train_q_error = LocalisationNetwork.Train_epoch(dataset=dataset, scene_info=scene_info, datagen=datagen,
                                                                      model=global_pose_network, quickTrain=False)
    # time.sleep(1)
    if ((epoch % epochs_per_result) == 0):
        test_xyz_error, test_q_error = LocalisationNetwork.Test_epoch(dataset=dataset, scene_info=scene_info, datagen=datagen, model=global_pose_network,
                                                  quickTest=False, getPrediction=False)
        print("Testing: [test_xyz_error,test_q_error] = [", test_xyz_error, ", ", test_q_error, "]", sep='')
        file1 = open("Results.txt", "a")
        file1.write(
            "%s,%s,%s,%s,%s\n" % (result_index, train_xyz_error, test_xyz_error, train_q_error, test_q_error))
        file1.close()
        xs.append(result_index)
        xyz_avg_error.append(test_xyz_error)
        q_avg_error.append(test_q_error)
        result_index += epochs_per_result
        # update_graph(xs,xyz_avg_error, q_avg_error)

    if (epoch == 10):
        LocalisationNetwork.Test_epoch(dataset=dataset, scene_info=scene_info, datagen=datagen, model=global_pose_network,
                                                  quickTest=False, getPrediction=True)

print("Finished Successfully")