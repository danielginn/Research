import sys
import os
from keras.preprocessing.image import load_img
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
from scipy.spatial.transform import Rotation as R
import json
from keras.callbacks import CSVLogger
import keras.backend as K
import tensorflow as tf
import tensorflow_probability as tfp

# Name: loadImages
# Inputs:
# - dataset (String): Name of Dataset used. Options are only 'NUbotsSoccerField1', 'NUbotsSoccerField2', or '7scenes'
# - data_purpose (String): Options are 'train' or 'test' only.
# - scene_info (array of dictionaries) : Meta-data about the data to be imported.
# Returns:
# - images (4-dim numpy array): [number of images, height, width, channels(=3)]
# - xyz (2-dim numpy array): [number of images, x-y-z coordinates(=3)]
# - q (2-dim numpy array): [number of images, quaternian]
# System Exit Conditions:
# - 'data_purpose must be test or train': Triggers if data_purpose input was not 'train' or 'test'
# - 'Unknown dataset': Triggers if dataset input was not one of the known dataset options
def loadImages(dataset, data_purpose, scene_info):

    if (dataset == 'NUbotsField'):
        if (data_purpose == 'train'):
            numImages = scene_info.get('num_train_images')
        elif (data_purpose == 'test'):
            numImages = scene_info.get('num_test_images')
        else:
            sys.exit('data_purpose must be test or train')

        scale = 0.5
        height = 224/scale
        images = np.zeros((numImages, int(round(height)), int(round(height*1.25)), 3))
        xyz = np.zeros((numImages, 3))
        q = np.zeros((numImages, 4))
        image_index = 0
        path = ".\\NUbotsField\\{}\\".format(data_purpose)
        print(path)
        for r, d, f in os.walk(path):
            for file in f:
                if '.jpg' in file:
                    img = load_img(os.path.join(r, file))
                    img = img.resize((int(round(height*1.25)), int(round(height))), Image.ANTIALIAS)
                    images[image_index, :, :, :] = img_to_array(img)

                    json_filename = file[0:-4] + '.json'
                    with open(os.path.join(r,json_filename)) as f2:
                        json_data = json.load(f2)
                    xyz[image_index,:] = json_data['position']
                    q[image_index,:] = json_data['rotation']

                    image_index += 1




    elif (dataset == '7scenes'):
        #sequences?
        if (data_purpose == 'train'):
            numImages = scene_info.get('num_images') * len(scene_info.get('train_sequences'))
        elif (data_purpose == 'test'):
            numImages = scene_info.get('num_images') * len(scene_info.get('test_sequences'))
        else:
            sys.exit('data_purpose must be test or train')

        images = np.zeros((numImages, 256, 341, 3))
        xyz = np.zeros((numImages, 3))
        q = np.zeros((numImages, 4))
        image_index = 0

        if (data_purpose == 'train'):
            sequences = scene_info.get('train_sequences')
        else:
            sequences = scene_info.get('test_sequences')

        images_in_seq = scene_info.get('num_images')
        # load the image
        for seq in sequences:
            for i in range(images_in_seq):
                # Load in image
                imageFileName = "./7scenes/{}/seq-{}/frame-{}.color.png".format(scene_info.get('name'),str(seq).zfill(2),str(i).zfill(6))
                img = load_img(imageFileName)
                img = img.resize((341,256),Image.ANTIALIAS)
                images[image_index,:,:,:] = img_to_array(img)

                # Load in pose data
                poseFileName = "./7scenes/{}/seq-{}/frame-{}.pose.txt".format(scene_info.get('name'),str(seq).zfill(2),str(i).zfill(6))
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
                q[image_index,:] = r.as_quat()
                # Extract xyz from homogeneous Transform
                xyz[image_index,:] = homogeneousTransform[0:3,3]
                file_handle.close()
                image_index += 1
    else:
        sys.exit('Unknown dataset')

    return images,xyz,q

def center_crop(img, crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = crop_size
    x = (width-dx)//2 + 1
    y = (height-dy)//2 + 1
    return img[y:(y+dy), x:(x+dx), :]

# Following 2 functions copied from https://jkjung-avt.github.io/keras-image-cropping/
def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

# This function has been modified from source to remove references to yield
def crop_generator(batches, crop_length, isRandom):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    batch_crops = np.zeros((batches.shape[0], crop_length, crop_length, 3))
    for i in range(batches.shape[0]):
        if isRandom:
            batch_crops[i] = random_crop(batches[i], (crop_length, crop_length))
        else:
            batch_crops[i] = center_crop(batches[i], (crop_length, crop_length))
    return batch_crops

# Custom metric
def xyz_error(y_true,y_pred):
    xtrue = y_true[:,0]
    ytrue = y_true[:,1]
    ztrue = y_true[:,2]
    xpred = y_pred[:,0]
    ypred = y_pred[:,1]
    zpred = y_pred[:,2]
    xyz_error = K.sqrt(K.square(xtrue-xpred) + K.square(ytrue-ypred) +K.square(ztrue-zpred))

    median_error = tfp.stats.percentile(xyz_error,q=50, interpolation='midpoint')

    return median_error


def Train_epoch(dataset, scene_info, datagen, model, quickTrain):
    xyz_error_sum = 0
    q_error_sum = 0
    num_scenes = 0
    for scene in scene_info:
        x_train, y_xyz_train, y_q_train = loadImages(dataset, 'train', scene)
        datagen.fit(x_train)
        for j in range(len(x_train)):
            x_train[j, :, :, :] = datagen.standardize(x_train[j, :, :, :])

        if (dataset == '7scenes'):
            isRandomCrops = True
        else:
            isRandomCrops = False
        x_train = crop_generator(x_train, 224, isRandom=isRandomCrops)
        csv_train_logger = CSVLogger(filename='training.log',append=True)
        if isinstance(model.output, list):
            print("Is a multiple output...assuming 2 outputs")
            history = model.fit(x=x_train, y={'xyz': y_xyz_train, 'q': y_q_train}, batch_size=32, verbose=0, shuffle=True)
            xyz_error_sum += history.history["xyz_mean_absolute_error"][0]
            q_error_sum += history.history["q_mean_absolute_error"][0]
        else:
            print("Is a single output")
            print("x_train length:",len(x_train))

            y_train = np.zeros([len(x_train),7])
            y_train[:,0:3] = y_xyz_train
            y_train[:,3:7] = y_q_train
            history = model.fit(x=x_train, y=y_train, batch_size=32, verbose=0, shuffle=True)
            print(history.history)
            xyz_error_sum += history.history["xyz_error"][0] # These are incorrect results and must be removed
            q_error_sum += history.history["xyz_error"][0] # These are incorrect results and must be removed


        num_scenes += 1
        if (quickTrain):
            break
    return model, xyz_error_sum/num_scenes, q_error_sum/num_scenes

def Test_epoch(dataset, scene_info, datagen, model, quickTest, getPrediction):
    xyz_error_sum = 0
    q_error_sum = 0
    num_scenes = 0
    for scene in scene_info:
        x_test, y_xyz_test, y_q_test = loadImages(dataset, 'test', scene)
        datagen.fit(x_test)
        for i in range(len(x_test)):
            x_test[i, :, :, :] = datagen.standardize(x_test[i, :, :, :])
        x_test = crop_generator(x_test, 224, isRandom=False)
        csv_test_logger = CSVLogger(filename='testing.log', append=True)
        if isinstance(model.output, list): # multi outputs...assuming 2
            results = model.evaluate(x=x_test, y={'xyz': y_xyz_test, 'q': y_q_test}, verbose=0)
            xyz_error_sum += results[3]
            q_error_sum += results[4]
            if (getPrediction):
                [xyz_predictions, q_predictions] = model.predict(x_test)
                np.savetxt("xyz_predictions.txt", xyz_predictions)
                np.savetxt("q_predictions.txt", q_predictions)
                np.savetxt("y_xyz_true.txt", y_xyz_test)
                np.savetxt("y_q_true.txt", y_q_test)
        else: # single output
            y_test = np.zeros([len(x_test), 7])
            y_test[:, 0:3] = y_xyz_test
            y_test[:, 3:7] = y_q_test
            results = model.evaluate(x=x_test, y=y_test, verbose=0)
            print(results)
            xyz_error_sum += 1 # false results. Need to remove
            q_error_sum += 1 # false results. Need to remove
            if (getPrediction):
                predictions = model.predict(x_test)
                np.savetxt("y_pred.txt",predictions)
                np.savetxt("y_true.txt",y_test)
        num_scenes += 1
        if (quickTest):
            break
    return xyz_error_sum/num_scenes, q_error_sum/num_scenes


