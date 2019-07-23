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
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.preprocessing.image import load_img

# print(device_lib.list_local_devices())
def loadImages(numImages):
    x = np.zeros((numImages,480,640,3))
    y = np.zeros((numImages,4,4))
    # load the image
    for i in range(numImages):
        # Load in image
        imageFileName = "./7scenes/chess/test/frame-{}.color.png".format(str(i).zfill(6))
        img = load_img(imageFileName)
        x[i,:,:,:] = img_to_array(img)

        # Load in pose data
        poseFileName = "./7scenes/chess/test/frame-{}.pose.txt".format(str(i).zfill(6))
        file_handle = open(poseFileName, 'r')
        # Read in all the lines of your file into a list of lines
        lines_list = file_handle.readlines()
        # Extract dimensions from first line. Cast values to integers from strings.
        #cols, rows = (float(val) for val in lines_list[0].split())
        # Do a double-nested list comprehension to get the rest of the data into your matrix
        my_data = [[float(val) for val in line.split()] for line in lines_list[0:]]
        file_handle.close()

        print(my_data)

    return x,y

base_model = ResNet50(weights='imagenet',include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)  # **** Assuming 2D, with no arguments required
x = Dense(1024, activation='relu', name='fc1')(x)  # **** Assuming relu
xyz = Dense(3, activation='softmax', name='xyz')(x)  # **** Assuming softmax is the correct activation here
rtp = Dense(4, activation='softmax', name='rtp')(x)  # **** Assuming softmax (rho/theta/phi) and quaternians

global_pose_network = Model(inputs=base_model.input, outputs=[xyz, rtp])

global_pose_network.compile(optimizer='Adam',loss='mean_squared_error')
#global_pose_network.summary()


x_train, y_train = loadImages(2)
print("Finished Successfully")
#train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

#train_generator = train_datagen.flow_from_directory('./7scenes/chess/test/',
#                                                      target_size=(224,224),
#                                                      color_mode='rgb',
#                                                      batch_size=32,
#                                                      class_mode='categorical',
#                                                      shuffle=True)




