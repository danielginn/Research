import sys
import os
from keras.preprocessing.image import load_img
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
from scipy.spatial.transform import Rotation as R
import json

def loadImages(dataset, data_purpose, scene_info):

    if (dataset == 'NUbotsSoccerField1'):
        if (data_purpose == 'train'):
            numImages = scene_info.get('num_train_images')
        elif (data_purpose == 'test'):
            numImages = scene_info.get('num_test_images')
        else:
            sys.exit('data_purpose must be test or train')

        images = np.zeros((numImages, 256, 341, 3))
        xyz = np.zeros((numImages, 3))
        q = np.zeros((numImages, 4))
        image_index = 0
        path = "D:\\VLocNet++\\Research\\NUbotsDatasets\\NUbotsSoccerField1\\{}\\".format(data_purpose)
        print(path)
        for r, d, f in os.walk(path):
            for file in f:
                if '.jpg' in file:
                    img = load_img(os.path.join(r, file))
                    img = img.resize((341, 256), Image.ANTIALIAS)
                    images[image_index, :, :, :] = img_to_array(img)

                    json_filename = file[0:-4] + '.json'
                    with open(os.path.join(r,json_filename)) as f2:
                        jsondata = json.load(f2)
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


def Train_epoch(dataset, scene_info, datagen, model, quickTrain):
    xyz_error_sum = 0
    q_error_sum = 0
    num_scenes = 0
    for scene in scene_info:
        x_train, y_xyz_train, y_q_train = loadImages(dataset, 'train', scene)
        datagen.fit(x_train)
        for j in range(len(x_train)):
            x_train[j, :, :, :] = datagen.standardize(x_train[j, :, :, :])
        x_train = crop_generator(x_train, 224, isRandom=True)
        history = model.fit(x=x_train, y={'xyz': y_xyz_train, 'q': y_q_train}, batch_size=32, verbose=0, shuffle=True)
        xyz_error_sum += history.history["xyz_mean_absolute_error"][0]
        q_error_sum += history.history["q_mean_absolute_error"][0]
        num_scenes += 1
        if (quickTrain):
            break
    return model, xyz_error_sum/num_scenes, q_error_sum/num_scenes

def Test_epoch(dataset, scene_info, datagen, model, quickTest):
    xyz_error_sum = 0
    q_error_sum = 0
    num_scenes = 0
    for scene in scene_info:
        x_test, y_xyz_test, y_q_test = loadImages(dataset, 'test', scene)
        datagen.fit(x_test)
        for i in range(len(x_test)):
            x_test[i, :, :, :] = datagen.standardize(x_test[i, :, :, :])
        x_test = crop_generator(x_test, 224, isRandom=False)
        results = model.evaluate(x=x_test, y={'xyz': y_xyz_test, 'q': y_q_test}, verbose=0)
        print(results)
        xyz_error_sum += results[3]
        q_error_sum += results[4]
        num_scenes += 1
        if (quickTest):
            break
    return xyz_error_sum/num_scenes, q_error_sum/num_scenes