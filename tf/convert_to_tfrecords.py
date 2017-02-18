# deeperic

"""Converts images and labels to TFRecords file format."""

import csv
import glob
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import helper

tf.app.flags.DEFINE_string('directory', '../tfrec',
                           'Directory to write the '
                           'converted result')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, name):
  num_examples = labels.shape[0]
  #print('labels shape is ' + str(labels.shape[0]))
  if images.shape[0] != num_examples:
    raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  #print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())



def read_images(path):
  print('Reading images')
  images = []
  png_files_path = glob.glob(os.path.join(path, './', '*.[pP][nN][gG]'))
  for filename in png_files_path:
    #print('read file:' + filename)
    im = Image.open(filename)  # .convert("L")  # Convert to greyscale
    im = np.asarray(im, np.uint8)

    # get only images name, not path
    image_name = filename.split('/')[-1].split('.')[0]
    images.append([int(image_name), im])

  images = sorted(images, key=lambda image: image[0])
  
  images_only = [np.asarray(image[1], np.uint8) for image in images]  # Use unint8 or you will be !!!
  images_only = np.array(images_only)

  return images_only

def read_labels(path, filename):
  print('Reading labels')
  with open(os.path.join(path, filename), 'r') as dest_f:
    data_iter = csv.reader(dest_f)
    train_labels = [data for data in data_iter]

  # pre process labels to int
  train_labels = _label_to_int(train_labels)
  train_labels = np.array(train_labels, dtype=np.uint32)

  return train_labels


def _label_to_int(labels):
  chars = helper.char_list
  new_labels = []

  for label in labels:
    new_labels.append(chars.index(label[1]))
  return new_labels

def main(argv):

  train_images = read_images('../training-character/train')
  train_labels = read_labels('../training-character', 'trainLabels.csv')
  validation_images = read_images('../training-character/val')
  validation_labels = read_labels('../training-character', 'valLabels.csv')
  
  print('train shape:' + str(train_images.shape))
  print('validation shape:' + str(validation_images.shape))

  convert_to(train_images, train_labels, 'train')
  convert_to(validation_images, validation_labels, 'validation')



if __name__ == '__main__':
  tf.app.run()
