import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import ResNet50Modifications as ResNetMods
import LocalisationNetwork
import DatasetInfo
import ProcessDataset
#tf.get_logger().setLevel('INFO')
#tf.autograph.set_verbosity(1)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
print("ResNet50 model loaded...")
for layer in base_model.layers[:143]: #175 is the final Activation layer: Activation_49, #143 is another one too.
    layer.trainable = False
#    print(layer.name)

base_model = ResNetMods.additional_final_layers(base_model)
global_pose_network = base_model
global_pose_network.compile(optimizer=Adam(lr=1e-4,epsilon=1e-10),loss='mean_squared_error', metrics=['accuracy'])
#global_pose_network.compile(optimizer=Adam(lr=1e-4,epsilon=1e-10),loss='mean_squared_error', metrics=[LocalisationNetwork.xyz_error])


######################################################################
###############  Training  ###########################################
######################################################################
print('*****************************')
print('***** STARTING TRAINING *****')
print('*****************************')

list_ds = tf.data.Dataset.list_files("7scenes\\chess\\seq-01\\*.color.png", shuffle=False)

for e in list_ds.take(1):
    result = ProcessDataset.process_path(e)
labeled_ds = list_ds.map(ProcessDataset.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = ProcessDataset.prepare_for_training(ds=labeled_ds, batch_size=32, cache='7scenes\\')

global_pose_network.fit(x=train_ds, epochs=5, verbose=2, steps_per_epoch=31)

#image_batch, label_batch = next(iter(train_ds))



