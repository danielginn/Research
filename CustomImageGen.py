import keras.backend as K
import tensorflow_probability as tfp
import numpy as np
from keras.preprocessing.image import load_img
from keras.callbacks import Callback
from keras.preprocessing.image import img_to_array
from scipy.spatial.transform import Rotation as R
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf
import glob


def geo_loss(y_true, y_pred):
    x_diff = y_true[:, 0] - y_pred[:, 0]
    y_diff = y_true[:, 1] - y_pred[:, 1]
    z_diff = y_true[:, 2] - y_pred[:, 2]

    q1_diff = y_true[:, 3] - y_pred[:, 3]
    q2_diff = y_true[:, 4] - y_pred[:, 4]
    q3_diff = y_true[:, 5] - y_pred[:, 5]
    q4_diff = y_true[:, 6] - y_pred[:, 6]

    L_x = K.sqrt(K.square(x_diff) + K.square(y_diff) + K.square(z_diff))
    L_q = K.sqrt(K.square(q1_diff) + K.square(q2_diff) + K.square(q3_diff) + K.square(q4_diff))

    B = 1
    return L_x + B*L_q

def xyz_error(y_true, y_pred):
    x_diff = y_true[:, 0] - y_pred[:, 0]
    y_diff = y_true[:, 1] - y_pred[:, 1]
    z_diff = y_true[:, 2] - y_pred[:, 2]
    xyz_error = K.sqrt(K.square(x_diff) + K.square(y_diff) + K.square(z_diff))

    median_error = tfp.stats.percentile(xyz_error, q=50, interpolation='midpoint')

    return median_error


def q_error(y_true, y_pred):
    return tf_function(y_true, y_pred)


@tf.function(input_signature=[tf.TensorSpec(shape=[32, 4], dtype=tf.float32), tf.TensorSpec(shape=[32, 4], dtype=tf.float32)])
def tf_function(input1, input2):
    y = tf.numpy_function(quat_diff, [input1, input2], tf.float32)
    return y


def quat_diff(y_true, y_pred):
    R_true = R.from_quat(y_true)
    R_pred = R.from_quat(y_pred)
    R_diff = R_true.inv()*R_pred
    q_diff = R_diff.as_quat()

    lengths = np.sqrt(np.square(q_diff[:, 0]) + np.square(q_diff[:, 1]) + np.square(q_diff[:, 2]))
    angles = np.degrees(2 * np.arctan2(lengths, q_diff[:, 3]))
    for i in range(angles.shape[0]):
        if angles[i] > 180:
            angles[i] = 360 - angles[i]
    median_error = np.mean(angles)
    #median_error = tfp.stats.percentile(angles, q=50, interpolation='midpoint')
    return K.cast(median_error, dtype='float32')


def list_of_files(purpose):
    return [f for f in glob.glob(".\\7scenes\\*\\" + purpose + "\\*\\*.color.png")]


def center_crop(img, crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = crop_size
    x = (width-dx)//2 + 1
    y = (height-dy)//2 + 1
    return img[y:(y+dy), x:(x+dx), :]


def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


def crop_generator(image, crop_length, isRandom):
    if isRandom:
        image_crop = random_crop(image, (crop_length, crop_length))
    else:
        image_crop = center_crop(image, (crop_length, crop_length))
    return image_crop


def get_input(path):
    img_full = load_img(path)

    if np.char.startswith(path, ".\\7scenes"):
        img_resized = img_full.resize((341, 256), Image.ANTIALIAS)
        img_np = img_to_array(img_resized)
        cropped_image = crop_generator(img_np, 224, isRandom=True)
    elif np.char.startswith(path, ".\\NUbotsField"):
        scale = 0.5
        height = 224 / scale
        img_resized = img_full.resize((int(round(height * 1.25)), int(round(height))), Image.ANTIALIAS)
        img_np = img_to_array(img_resized)
        cropped_image = crop_generator(img_np, 224, isRandom=False)

    return cropped_image


def get_output(image_path):
    if np.char.startswith(image_path, ".\\7scenes"):
        xyzq = np.zeros(7)
        pose_path = image_path[:-9] + "pose.txt"
        file_handle = open(pose_path, 'r')

        # Read in all the lines of your file into a list of lines
        lines_list = file_handle.readlines()
        # Do a double-nested list comprehension to store as a Homogeneous Transform matrix
        homogeneous_transform_list = [[float(val) for val in line.split()] for line in lines_list[0:]]
        homogeneous_transform = np.zeros((4, 4))

        for j in range(4):
            homogeneous_transform[j, :] = homogeneous_transform_list[j]

        # Extract xyz from homogeneous Transform
        xyzq[0:3] = homogeneous_transform[0:3, 3]
        # Extract rotation from homogeneous Transform
        r = R.from_dcm(homogeneous_transform[0:3, 0:3])
        xyzq[3:7] = r.as_quat()

        file_handle.close()
        return xyzq

    elif np.char.startswith(image_path, ".\\NUbotsField"):
        pose_path = image_path[:-3] + "json"

    else:
        print("Unrecognised dataset")

    return 0


def image_generator(files, batch_size):
    i = 0
    while True:
        batch_start = max(((i+batch_size) % len(files))-batch_size, 0) # This makes sure a batch too close to the end is not chosen
        batch_end = batch_start + batch_size
        i += batch_size

        #Select files (paths/indices) for the batch
        #batch_paths = np.random.choice(files, batch_size)
        batch_paths = np.array(files[batch_start:batch_end])
        batch_input = []
        batch_output = []

        # Read in each input, perform preprocessing and get labels
        for input_path in batch_paths:
            input = get_input(input_path)
            output = get_output(input_path)

            batch_input += [input]
            batch_output += [output]

        # Return a tuple of (input, output) to feed the network
        datagen = ImageDataGenerator()

        batch_x = np.array(batch_input)
        batch_x_st = datagen.standardize(batch_x)
        batch_y = np.array(batch_output)
        yield (batch_x_st, batch_y)
