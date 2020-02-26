import pandas as pd
import re
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras
import scipy
import matplotlib.pyplot as plt
from matplotlib import style
import sys
import time
import random
from matplotlib.pyplot import draw
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, concatenate, Reshape, Input, Conv2D, Concatenate, BatchNormalization, Add, Dropout
from tensorflow.keras.initializers import VarianceScaling, Ones
from tensorflow.keras.applications import ResNet50
import datetime
from tensorflow.python.keras.callbacks import TensorBoard
#from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import load_img
from scipy.spatial.transform import Rotation as R
import ResNet50Modifications as ResNetMods
import LocalisationNetwork
import DatasetInfo
from tensorboard import program

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

#with tf.device('/device:GPU:0'):
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
print("ResNet50 model loaded...")
for layer in base_model.layers[:143]: #175 is the final Activation layer: Activation_49, #143 is another one too.
    layer.trainable = False
#    print(layer.name)

base_model = ResNetMods.additional_final_layers(base_model)
global_pose_network = base_model

# Setting up tensorboard. To run, type 'tensorboard --logdir=logs/' into pycharm's terminal
t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
fit_log_dir="logs\\" + t + "\\fit"
eval_log_dir="logs\\" + t + "\\eval"
fit_tensorboard = TensorBoard(log_dir=fit_log_dir, profile_batch=0)
eval_tensorboard = TensorBoard(log_dir=eval_log_dir, profile_batch=0)


global_pose_network.compile(optimizer=Adam(lr=1e-4,epsilon=1e-10),loss='mean_squared_error', metrics=[LocalisationNetwork.median_xyz_error])
#global_pose_network.summary()

dataset = '7scenes' # Can be: 7scenes, NUbotsSoccerField1, NUbotsSoccerField2
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
file1 = open("Results.txt", "w")
# Base-line accuracy
test_xyz_error, test_q_error = LocalisationNetwork.Test_epoch(dataset=dataset, scene_info=scene_info, datagen=datagen, model=global_pose_network,
                                          quickTest=True, getPrediction=False, callbacks=eval_tensorboard)
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
                                                                      model=global_pose_network, quickTrain=True, callbacks=fit_tensorboard)
    # time.sleep(1)
    if ((epoch % epochs_per_result) == 0):
        test_xyz_error, test_q_error = LocalisationNetwork.Test_epoch(dataset=dataset, scene_info=scene_info, datagen=datagen, model=global_pose_network,
                                                  quickTest=True, getPrediction=False, callbacks=eval_tensorboard)
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

    #if (epoch == 10):
    #    LocalisationNetwork.Test_epoch(dataset=dataset, scene_info=scene_info, datagen=datagen, model=global_pose_network,
    #                                              quickTest=True, getPrediction=True)

print("Finished Successfully")