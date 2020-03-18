import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import math
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import ResNet50Modifications as ResNetMods
import LocalisationNetwork
import DatasetInfo
import ProcessDataset
import CustomMethods
#tf.get_logger().setLevel('INFO')
#tf.autograph.set_verbosity(1)



######################################################################
###############  Load Images  ########################################
######################################################################
list_train_ds = tf.data.Dataset.list_files("7scenes\\*\\train\\seq-*\\*.color.png", shuffle=True)
list_test_ds = tf.data.Dataset.list_files("7scenes\\*\\test\\seq-*\\*.color.png", shuffle=True)

num_train_images = 0
for f in list_train_ds:
    num_train_images += 1
print("total training images:",num_train_images)
steps_per_epoch_train = math.ceil(num_train_images/32)

num_test_images = 0
for f in list_test_ds:
    num_test_images += 1
print("total test images:",num_test_images)
steps_per_epoch_test = math.ceil(num_train_images/32)

######################################################################
###############  Model  ##############################################
######################################################################

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
print("ResNet50 model loaded...")
for layer in base_model.layers[:143]: #175 is the final Activation layer: Activation_49, #143 is another one too.
    layer.trainable = False
#    print(layer.name)

if num_train_images > num_test_images:
    median_array_size = num_train_images
else:
    median_array_size = num_test_images

base_model = ResNetMods.additional_final_layers(base_model)
global_pose_network = base_model
global_pose_network.compile(optimizer=Adam(lr=1e-4,epsilon=1e-10),loss='mean_squared_error', metrics=[CustomMethods.Mean_XYZ_Error(batch=32)])


######################################################################
###############  Preprocessing Images  ###############################
######################################################################

labeled_train_ds = list_train_ds.map(ProcessDataset.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
labeled_test_ds = list_test_ds.map(ProcessDataset.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = ProcessDataset.prepare_for_training(ds=labeled_train_ds, batch_size=32, cache='7scenes\\Train\\', shuffle_buffer_size=num_train_images)
test_ds = ProcessDataset.prepare_for_training(ds=labeled_test_ds, batch_size=32, cache='7scenes\\Test\\', shuffle_buffer_size=num_test_images)


######################################################################
###############  Training  ###########################################
######################################################################
print('*****************************')
print('***** STARTING TRAINING *****')
print('*****************************')
global_pose_network.fit(x=train_ds, epochs=30, verbose=2, steps_per_epoch=steps_per_epoch_train, validation_data=test_ds, validation_steps=steps_per_epoch_test, validation_freq=5)





