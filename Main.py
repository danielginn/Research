import pandas as pd
import numpy as np
import os
import keras
import scipy
import matplotlib as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.preprocessing.image import load_img
from scipy.spatial.transform import Rotation as R

# print(device_lib.list_local_devices())
def loadImages(numImages):
    images = np.zeros((numImages,480,640,3))
    xyz = np.zeros((numImages,3))
    q = np.zeros((numImages,4))
    # load the image
    for i in range(numImages):
        # Load in image
        imageFileName = "./7scenes/chess/test/frame-{}.color.png".format(str(i).zfill(6))
        img = load_img(imageFileName)
        images[i,:,:,:] = img_to_array(img)

        # Load in pose data
        poseFileName = "./7scenes/chess/test/frame-{}.pose.txt".format(str(i).zfill(6))
        file_handle = open(poseFileName, 'r')
        # Read in all the lines of your file into a list of lines
        lines_list = file_handle.readlines()
        # Do a double-nested list comprehension to store as a Homogeneous Transform matrix
        homogeneousTransformList = [[float(val) for val in line.split()] for line in lines_list[0:]]
        homogeneousTransform = np.zeros((4,4))

        for j in range(4):
            homogeneousTransform[j,:] = homogeneousTransformList[j]

        # Extract rotation from homogeneous Transform
        r = R.from_dcm(homogeneousTransform[0:3,0:3])
        q = r.as_quat()
        # Extract xyz from homogeneous Transform
        xyz[i,:] = homogeneousTransform[0:3,3]

        file_handle.close()
    return images,xyz,q

x_train, y_xyz_train, y_q_train = loadImages(2)

base_model = ResNet50(weights='imagenet',include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)  # **** Assuming 2D, with no arguments required
x = Dense(1024, activation='relu', name='fc1')(x)  # **** Assuming relu
xyz = Dense(3, activation='softmax', name='xyz')(x)  # **** Assuming softmax is the correct activation here
q = Dense(4, activation='softmax', name='q')(x)  # **** Assuming softmax (rho/theta/phi) and quaternians

global_pose_network = Model(inputs=base_model.input, outputs=[xyz, q])

global_pose_network.compile(optimizer='Adam',loss='mean_squared_error')
#global_pose_network.summary()

datagen = ImageDataGenerator()

global_pose_network.fit(x=x_train, y=[y_xyz_train, y_q_train], batch_size=32, epochs=1, verbose=1, shuffle=False, steps_per_epoch=None)

print("Finished Successfully")