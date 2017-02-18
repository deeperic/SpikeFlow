#!/usr/bin/env python
# -*- coding: UTF-8 -*-



#deeperic

import os.path
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import glob
from PIL import Image
import sys

import spikeflow
import ocr_model
import helper

#import input_data

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_epochs', 50000, 'Number of epochs to run trainer.')

flags.DEFINE_integer('batch_size', 32, 'Batch size.')


IMAGE_PIXELS = 32 * 32 * 3


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded ckpt in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test ckpt sets.
  # batch_size = -1
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         IMAGE_PIXELS))
                                                         
  labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)

  return images_placeholder, labels_placeholder

  
def test(argv):

  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
    
    test_images, test_labels = spikeflow.inputs_test(filename=argv[1], batch_size=FLAGS.batch_size,
                                    num_epochs=FLAGS.num_epochs,
                                    num_threads=5, imshape=[32, 32, 3])

    
    # Build a Graph that computes the logits predictions from the inference model.
    logits = ocr_model.inference(images_placeholder)

    # Create a saver.
    saver = tf.train.Saver()

    init = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

    # Start running operations on the Graph.
    NUM_CORES = 5  # Choose how many cores to use.
    
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                   intra_op_parallelism_threads=NUM_CORES))
                               
    sess.run(init)

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                                    
    #print('Model:' + argv[0])
    saver.restore(sess, argv[0])
    #print('Model Restored!')
    
        
    images_test_r, labels_test_r = sess.run([test_images, test_labels])
        
    val_feed_test = {images_placeholder: images_test_r,
                    labels_placeholder: labels_test_r}
                    
    acc_r = sess.run(tf.argmax(logits,1), feed_dict=val_feed_test)
        
    chars = helper.char_list_chinese
          
    print('The character is: ' + str(chars[acc_r[0]]))
    
    #print predictions
    prediction=tf.argmax(logits,1)
    print "Predictions:", prediction.eval(feed_dict=val_feed_test, session=sess)[0]

    #print probabilities
    probabilities=logits
    print "Probabilities:", probabilities.eval(feed_dict=val_feed_test, session=sess)[0]
      
    coord.request_stop()
    coord.join(threads)
    sess.close()



def main(argv=None):
#python test_one_char.py {model_name} {image_name}
  test(sys.argv[1:]) #predict

if __name__ == '__main__':
  tf.app.run()
