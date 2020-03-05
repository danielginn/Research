import tensorflow as tf
import numpy as np
#import io
from scipy.spatial.transform import Rotation as R

IMG_WIDTH = 341
IMG_HEIGHT = 256

def get_xyzq(file_path):
  xyzq = np.zeros(7)
  parts = tf.strings.split(file_path, '.')
  poseFileName = tf.strings.join([parts[0], '.pose.txt'])

  input = tf.io.read_file(poseFileName)
  transform = convert_string_to_xyzq(input)

  return transform

def decode_img(img):
  img = tf.image.decode_png(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

def process_path(file_path):
  xyzq = get_xyzq(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)

  return img,xyzq

def prepare_for_training(ds, batch_size=32, cache=True, shuffle_buffer_size=1000):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.map(final_crop)
  print(ds)
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(batch_size)

  # `prefetch` lets the dataset fetch batches in the background while the model is training
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return ds

def final_crop(img, xyzq):
  img = tf.image.random_crop(img, [224, 224, 3])
  xyzq = tf.reshape(xyzq, [7,1])
  return img, xyzq

def convert_string_to_xyzq(str):
  elements = tf.strings.split(str, '\r\n')
  row1 = tf.strings.split(elements[0], '')
  row2 = tf.strings.split(elements[1], '')
  row3 = tf.strings.split(elements[2], '')
  row4 = tf.strings.split(elements[3], '')

  numbers1 = tf.strings.to_number(row1)
  numbers2 = tf.strings.to_number(row2)
  numbers3 = tf.strings.to_number(row3)
  numbers4 = tf.strings.to_number(row4)
  matrix = tf.reshape(tf.concat([numbers1,numbers2,numbers3,numbers4],0),[4,4])

  # Extract rotation from homogeneous Transform
  xyzq = tf_function(matrix)
  return xyzq


def my_H_transform(x):
  r = R.from_dcm(x[0:3,0:3])
  q = r.as_quat().astype(np.float32)
  xyz = x[0:3,3]
  xyzq = np.concatenate((xyz,q))
  return xyzq

@tf.function(input_signature=[tf.TensorSpec(shape=[4,4], dtype=tf.float32)])
def tf_function(input):
  y = tf.numpy_function(my_H_transform, [input], tf.float32)
  return y
