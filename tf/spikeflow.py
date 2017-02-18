#deeperic

import convert_to_tfrecords
import reader


from numpy.random import shuffle
import tensorflow as tf



def inputs_test_files(filename, batch_size, num_epochs2, num_threads,
           imshape, num_examples_per_epoch=128):
  """Reads input tfrecord file num_epochs times. Use it for validation.

  Args:
    filename: The path to the .tfrecords file to be read
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input ckpt, or 0/None to
       train forever.
    num_threads: Number of reader workers to enqueue
    imshape: The shape of image in the format
    num_examples_per_epoch: Number of images to use per epoch

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """


  tf.local_variables_initializer()
  
  if not num_epochs2:
    num_epochs2 = None

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      #[filename], num_epochs=num_epochs2, name='string_input_producer')
      
      #['../character-data/test/170.png',
      #'../character-data/test/214.png',
      #'../character-data/test/2579.png'],
      ['../project/pricesplit_/0.png',
      '../project/pricesplit_/1.png',
      '../project/pricesplit_/2.png',
      '../project/pricesplit_/3.png',
      '../project/pricesplit_/4.png',
      '../project/pricesplit_/5.png',
      '../project/pricesplit_/6.png',
      '../project/pricesplit_/7.png'],
       num_epochs=num_epochs2, 
       shuffle=False,#must be false otherwise will shuffle everytime
       name='string_input_producer')



    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = reader.read_and_decode_eric(filename_queue, imshape, normalize=True)
	
    # Convert from [0, 255] -> [-0.5, 0.5] floats. The normalize param in read_and_decode will do the same job.
    # image = tf.cast(image, tf.float32)
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    # Ensure that the random shuffling has good mixing properties.
    #min_fraction_of_examples_in_queue = 0.4
    #min_queue_examples = int(num_examples_per_epoch *
    #                         min_fraction_of_examples_in_queue)

    #images, sparse_labels = tf.train.shuffle_batch(
    #  [image, label], batch_size=batch_size, num_threads=num_threads,
    #  capacity=min_queue_examples + 3 * batch_size, enqueue_many=False,
    #  # Ensures a minimum amount of shuffling of examples.
    #  min_after_dequeue=min_queue_examples, name='batching_shuffling')
      
    #print('erieric')
      
    images, sparse_labels = tf.train.batch(
      [image, label], batch_size=batch_size, num_threads=num_threads)

    return images, sparse_labels
    
def inputs_test(filename, batch_size, num_epochs, num_threads,
           imshape, num_examples_per_epoch=128):
 
  tf.local_variables_initializer()
  
  if not num_epochs:
    num_epochs = None

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      [filename], num_epochs=num_epochs, name='string_input_producer')
      
    image, label = reader.read_and_decode_wholefile(filename_queue, imshape, normalize=True)
	
    images, sparse_labels = tf.train.batch([image, label], batch_size=batch_size)  
 
    return images, sparse_labels
    
  

def inputs(filename, batch_size, num_epochs2, num_threads,
           imshape, num_examples_per_epoch=128):
  """Reads input tfrecord file num_epochs times. Use it for validation.

  Args:
    filename: The path to the .tfrecords file to be read
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input ckpt, or 0/None to
       train forever.
    num_threads: Number of reader workers to enqueue
    imshape: The shape of image in the format
    num_examples_per_epoch: Number of images to use per epoch

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """


  tf.local_variables_initializer()
  
  if not num_epochs2:
    num_epochs2 = None

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      [filename], num_epochs=num_epochs2, name='string_input_producer')

    # Even when reading in multiple threads, share the filename
    # queue.
    #image, label = reader.read_and_decode(filename_queue, imshape, normalize=True, flatten=False)
    image, label = reader.read_and_decode(filename_queue, imshape, normalize=True)

    # Convert from [0, 255] -> [-0.5, 0.5] floats. The normalize param in read_and_decode will do the same job.
    # image = tf.cast(image, tf.float32)
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    images, sparse_labels = tf.train.shuffle_batch(  
      [image, label], batch_size=batch_size, num_threads=num_threads,
      capacity=min_queue_examples + 3 * batch_size, enqueue_many=False,
      # Ensures a minimum amount of shuffling of examples.
      min_after_dequeue=min_queue_examples, name='batching_shuffling')
      
    return images, sparse_labels

def _random_brightness_helper(image):
  return tf.image.random_brightness(image, max_delta=63)
    
def _random_contrast_helper(image):  
  return tf.image.random_contrast(image, lower=0.2, upper=1.8)

def distorted_inputs(filename, batch_size, num_epochs, num_threads,
                     imshape, num_examples_per_epoch=128, flatten=True):
  """Construct distorted input for training using the Reader ops.

  Raises:
    ValueError: if no data_dir

  Args:
    filename: The name of the file containing the images
    batch_size: The number of images per batch
    num_epochs: The number of epochs passed to string_input_producer
    num_threads: The number of threads passed to shuffle_batch
    imshape: Shape of image in [height, width, n_channels] format
    num_examples_per_epoch: Number of images to use per epoch
    flatten: Whether to flatten image after image transformations

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  
  tf.local_variables_initializer()
  tf.global_variables_initializer()
  
  if not num_epochs:
    num_epochs = None

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
      [filename], num_epochs=num_epochs, name='string_DISTORTED_input_producer')

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = reader.read_and_decode(filename_queue, imshape)

    # Reshape to imshape as distortion methods need this shape
    image = tf.reshape(image, imshape)

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Removed random_crop in new TensorFlow release.
    # Randomly crop a [height, width] section of the image.
    # distorted_image = tf.image.random_crop(image, [height, width])
    #
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(image)
    #
    # Randomly apply image transformations in random_functions list
    random_functions = [_random_brightness_helper, _random_contrast_helper]
    shuffle(random_functions)
    for fcn in random_functions:
      distorted_image = fcn(distorted_image)

    # # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    if flatten:
      num_elements = 1
      for i in imshape: num_elements = num_elements * i
      image = tf.reshape(float_image, [num_elements])
    else:
      image = float_image

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                   batch_size=batch_size,
                                                   num_threads=num_threads,
                                                   capacity=min_queue_examples + 3 * batch_size,
                                                   enqueue_many=False,
                                                   # Ensures a minimum amount of shuffling of examples.
                                                   min_after_dequeue=min_queue_examples,
                                                   name='batching_shuffling_distortion')

  return images, sparse_labels


