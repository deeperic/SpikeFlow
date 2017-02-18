#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#deeperic

"""Training the model"""


import os.path
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import glob
from PIL import Image

import spikeflow
import ocr_model

#import input_data

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 50000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', '../tmp/my-model',
                           """Directory where to write model proto """
                           """ to import in c++""")
                           
tf.app.flags.DEFINE_string('train_dir', '../tmp/log',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
                            
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoint',
                           """Directory where to read/write model checkpoints.""")

# Parameters
display_step = 100
val_step = 500
save_step = 500
PIXEL_DIM = 32
IMAGE_PIXELS = PIXEL_DIM * PIXEL_DIM * 3
NEW_LINE = '\n'



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

  
def train(continue_from_pre = False):
  
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
    
    #read the TF Records
    images, labels = spikeflow.inputs(filename='../tfrec/train.tfrecords', batch_size=FLAGS.batch_size,
                                      num_epochs2=FLAGS.num_epochs,
                                      num_threads=5, imshape=[PIXEL_DIM, PIXEL_DIM, 3])
                                      
                                      
    val_images, val_labels = spikeflow.inputs(filename='../tfrec/validation.tfrecords', batch_size=FLAGS.batch_size,
                                    num_epochs2=FLAGS.num_epochs,
                                    num_threads=5, imshape=[PIXEL_DIM, PIXEL_DIM, 3])
                                    
    # Build a Graph that computes the logits predictions from the inference model.
    logits = ocr_model.inference(images_placeholder)

    # Calculate loss.
    loss = ocr_model.loss(logits, labels_placeholder)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = ocr_model.training(loss, global_step)

    # Calculate accuracy #
    acc, n_correct = ocr_model.evaluation(logits, labels_placeholder)

    # Create a saver.
    saver = tf.train.Saver()

    tf.summary.scalar('Acc', acc)
    tf.summary.scalar('Loss', loss)
 
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.

    init = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

    # Start running operations on the Graph.
    NUM_CORES = 5  # Choose how many cores to use.
    
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                   intra_op_parallelism_threads=NUM_CORES))
                   
    sess.run(init)

    # Write all terminal output results here
    val_f = open("../tmp/val.txt", "ab")

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                            graph=sess.graph)

    # Export graph to import it later in c++
    # tf.train.write_graph(sess.graph, FLAGS.model_dir, 'train.pbtxt') # TODO: uncomment to get graph and use in c++

    #reload previous saved check points
    continue_from_pre = False

    if continue_from_pre:
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
      print ckpt.model_checkpoint_path
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Session Restored!')

    try:
      while not coord.should_stop():

        for step in xrange(FLAGS.max_steps):

          images_r, labels_r = sess.run([images, labels])
          images_val_r, labels_val_r = sess.run([val_images, val_labels])

          train_feed = {images_placeholder: images_r,
                          labels_placeholder: labels_r}

          val_feed = {images_placeholder: images_val_r,
                        labels_placeholder: labels_val_r}

          start_time = time.time()

          _, loss_value = sess.run([train_op, loss], feed_dict=train_feed)
          duration = time.time() - start_time

          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

          if step % display_step == 0:
            num_examples_per_step = FLAGS.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('%s: step %d, loss = %.6f (%.1f examples/sec; %.3f '
                            'sec/batch)')
            print_str_loss = format_str % (datetime.now(), step, loss_value,
                                             examples_per_sec, sec_per_batch)
            print (print_str_loss)
            val_f.write(print_str_loss + NEW_LINE)
            summary_str = sess.run([summary_op], feed_dict=train_feed)
            summary_writer.add_summary(summary_str[0], step)

          if step % val_step == 0:
            acc_value, num_corroect = sess.run([acc, n_correct], feed_dict=train_feed)

            format_str = '%s: step %d,  train acc = %.2f, n_correct= %d'
            print_str_train = format_str % (datetime.now(), step, acc_value, num_corroect)
            val_f.write(print_str_train + NEW_LINE)
            print (print_str_train)

          # Save the model checkpoint periodically.
          if step % save_step == 0 or (step + 1) == FLAGS.max_steps:
            val_acc_r, val_n_correct_r = sess.run([acc, n_correct], feed_dict=val_feed)

            frmt_str = '%s: step %d, Val Acc = %.2f, num correct = %d'
            print_str_val = frmt_str % (datetime.now(), step, val_acc_r, val_n_correct_r)
            val_f.write(print_str_val + NEW_LINE)
            print(print_str_val)

            checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
            savedpath = saver.save(sess, checkpoint_path, global_step=step)
            print('model saved: ' + savedpath)


    except tf.errors.OutOfRangeError:
      print ('Done training -- epoch limit reached')

    finally:
      # When done, ask the threads to stop.
      val_f.write(NEW_LINE +
                    NEW_LINE +
                    '############################ FINISHED ############################' +
                    NEW_LINE)
      val_f.close()
      coord.request_stop()

      # Wait for threads to finish.
    coord.join(threads)
    sess.close()

  
def main(argv=None):
  train() 

if __name__ == '__main__':
  tf.app.run()

