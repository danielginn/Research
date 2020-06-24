import keras.backend as K
import tensorflow_probability as tfp
import numpy as np
from keras.preprocessing.image import load_img
from keras.callbacks import Callback
from keras.metrics import Metric
from keras.preprocessing.image import img_to_array
from scipy.spatial.transform import Rotation as R
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf
import glob


class MyMetrics(Callback):
    def __init__(self, val_data, batches, batch_size=32):
        self.validation_data = val_data
        self.batch_size = batch_size
        self.batches = batches

    def on_test_begin(self, logs=None):
        self.errors = []

    #def on_epoch_end(self, epoch, logs=None):
    def on_test_end(self, logs=None):
        total =  self.batches * self.batch_size
        val_pred = np.zeros((total,7))
        val_true = np.zeros((total,7))

        for batch in range(self.batches):
            xVal, yVal = next(self.validation_data)
            val_pred[batch * self.batch_size : (batch+1) * self.batch_size, :] = np.asarray(self.model.predict(xVal))
            val_true[batch * self.batch_size : (batch+1) * self.batch_size, :] = yVal

        x_diff = val_true[:, 0] - val_pred[:, 0]
        y_diff = val_true[:, 1] - val_pred[:, 1]
        z_diff = val_true[:, 2] - val_pred[:, 2]
        xyz_error = np.sqrt(np.square(x_diff)+np.square(y_diff)+np.square(z_diff))

        for e in xyz_error:
            self.errors.append(e)

    def get_median(self):
        self.median = np.median(self.errors)
        return self.median

    def get_outliers(self):
        std = np.std(self.errors)
        outliers = []
        i = 0
        for e in self.errors:
            i += 1
            if e > (self.median + 4*std):
                outliers.append((e, i))

        return outliers

    def get_all_errors(self):
        return self.errors






def geo_loss(y_true, y_pred):
    x_diff = y_true[:, 0] - y_pred[:, 0]
    y_diff = y_true[:, 1] - y_pred[:, 1]
    z_diff = y_true[:, 2] - y_pred[:, 2]

    #q_pred = K.l2_normalize(y_pred[:, 3:7], axis=1)
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


def list_of_files(folder,purpose):
    #return [f for f in glob.glob(".\\7scenes\\*\\" + purpose + "\\*\\*.color.png")]
    #return [f for f in glob.glob(".\\7scenes\\"+ folder +"\\" + purpose + "\\*\\*.color.png")]
    return [f for f in glob.glob(".\\7scenes\\redkitchen\\" + purpose + "\\" + folder + "\\*.color.png")]


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

def get_inputs(path):
    img_full = load_img(path)
    img_resized = img_full.resize((341, 256), Image.ANTIALIAS)
    img_np = img_to_array(img_resized)
    cropped_image = crop_generator(img_np, 224, isRandom=True)
    prev_num = int(path[-10]) - 1
    pose_path = path[:-11] + str(prev_num) + path[-9:]
    print(path)
    print(pose_path)
    print("-----------------")
    xyzq = get_output(pose_path)

    return cropped_image, xyzq


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

        #if np.char.startswith(image_path, ".\\7scenes\\c"):
        #    xyzq[7] = 100000
        #elif np.char.startswith(image_path, ".\\7scenes\\f"):
        #    xyzq[7] = 200000
        #elif np.char.startswith(image_path, ".\\7scenes\\h"):
        #    xyzq[7] = 300000
        #elif np.char.startswith(image_path, ".\\7scenes\\o"):
        #    xyzq[7] = 400000
        #elif np.char.startswith(image_path, ".\\7scenes\\p"):
        #    xyzq[7] = 500000
        #elif np.char.startswith(image_path, ".\\7scenes\\r"):
        #    xyzq[7] = 600000
        #elif np.char.startswith(image_path, ".\\7scenes\\s"):
        #    xyzq[7] = 700000

        #xyzq[7] += int(image_path[-25:-23])*1000
        #xyzq[7] += int(image_path[-13:-10])

        file_handle.close()
        return xyzq

    elif np.char.startswith(image_path, ".\\NUbotsField"):
        pose_path = image_path[:-3] + "json"

    else:
        print("Unrecognised dataset")

    return 0


def image_generator(files, batch_size, feedback_loop, is_random, steps):

    if feedback_loop:
        while True:
            #Select files (paths/indices) for the batch
            batch_paths = np.random.choice(files, batch_size)
            batch_image_input = []
            batch_pose_input = []
            batch_output = []
            # Read in each input, perform preprocessing and get labels
            for input_path in batch_paths:
                image_input, pose_input = get_inputs(input_path)
                output = get_output(input_path)

                batch_image_input += [image_input]
                batch_pose_input += [pose_input]
                batch_output += [output]

            # Return a tuple of (input, output) to feed the network
            datagen = ImageDataGenerator()

            batch_x = {"image_input":datagen.standardize(np.array(batch_image_input)), "pose_input":np.array(batch_pose_input)}
            batch_y = np.array(batch_output)
            yield (batch_x, batch_y)

    else:
        i = 0
        while True:
            # Select files (paths/indices) for the batch
            if is_random:
                batch_paths = np.random.choice(files, batch_size)
            else:
                batch_start = max(((i + batch_size) % len(files)) - batch_size, 0)  # This makes sure a batch too close to the end is not chosen
                batch_end = batch_start + batch_size
                i += batch_size
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

            batch_x = datagen.standardize(np.array(batch_input))
            batch_y = np.array(batch_output)
            yield (batch_x, batch_y)