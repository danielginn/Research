import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import ResNet50Modifications as ResNetMods
import LocalisationNetwork
import DatasetInfo
import ProcessDataset

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
print("ResNet50 model loaded...")
for layer in base_model.layers[:143]: #175 is the final Activation layer: Activation_49, #143 is another one too.
    layer.trainable = False
#    print(layer.name)

base_model = ResNetMods.additional_final_layers(base_model)
global_pose_network = base_model
global_pose_network.compile(optimizer=Adam(lr=1e-4,epsilon=1e-10),loss='mean_squared_error', metrics=['accuracy'])
#global_pose_network.compile(optimizer=Adam(lr=1e-4,epsilon=1e-10),loss='mean_squared_error', metrics=[LocalisationNetwork.xyz_error])

#dataset = '7scenes' # Can be: 7scenes, NUbotsSoccerField1, NUbotsSoccerField2
#scene_info = DatasetInfo.GetDatasetInfo(dataset)

######################################################################
###############  Training  ###########################################
######################################################################
print('*****************************')
print('***** STARTING TRAINING *****')
print('*****************************')

list_ds = tf.data.Dataset.list_files("7scenes\\chess\\seq-01\\*.color.png", shuffle=False)
labeled_ds = list_ds.map(ProcessDataset.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = ProcessDataset.prepare_for_training(ds=labeled_ds, batch_size=32, cache='7scenes\\')

image_batch, label_batch = next(iter(train_ds))



