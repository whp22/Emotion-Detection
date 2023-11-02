from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf


from datasets import dataset_utils

#====================================================DEFINR YOUR ARGUMENTS==========================================================
from datasets.dataset_utils import image_to_tfexample

flags = tf.app.flags

flags.DEFINE_string('dataset_dir', '/home/joy/Desktop/YU/fer2013/valid_folder', 'String: The dataset directory to where your store your images')

flags.DEFINE_float('validation_ratio', 0, 'Float: The proportion of examples in the dataset to be used for validation')

flags.DEFINE_integer('number_of_shards', 1, 'Int: Number of shards to split the TFRecord files')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', 'validation', 'String: The output filename to name your TFRecord file')

#Choice of which folder to convert to TFRecord file
flags.DEFINE_string('folder', 'validation', 'String: The name of train or validation folder')

FLAGS = flags.FLAGS

#======================================================CONVERSION UTILS=============================================================
#Create an image reader object for easy reading of the jpeg images
class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=1)  #!!!!REVISE CHANNELS FOR GRAY SCALE IMAGES

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 1
    return image

def _get_filenames_and_classes(dataset_dir, folder):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  #flower_root = os.path.join(dataset_dir, 'flower_photos')
  images_root = os.path.join(dataset_dir, '%s' %folder)
  directories = []
  class_names = []
  for filename in os.listdir(images_root):
    path = os.path.join(images_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)

def _get_dataset_filename(dataset_dir, split_name, shard_id, _NUM_SHARDS):
  output_filename = 'fer2013_%s_%01d-of-%01d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, _NUM_SHARDS):
  """Converts the given filenames to a TFRecord dataset.
  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id, _NUM_SHARDS = _NUM_SHARDS)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            #image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()         #!!!!'r' vs 'rb'???
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = image_to_tfexample(
                image_data, 'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

'''
def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'flower_photos')
  tf.gfile.DeleteRecursively(tmp_dir)
'''

def _dataset_exists(dataset_dir, _NUM_SHARDS):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id, _NUM_SHARDS = _NUM_SHARDS)
      if not tf.gfile.Exists(output_filename):
        return False
  return True

#==========================================================CONVERSION===============================================================
def main():
    #============================================CHECK===============================================
    # Check if there is a tfrecord_filename entered
    if not FLAGS.tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

    # Check if there is a dataset directory entered
    if not FLAGS.dataset_dir:
        raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

    # If the TFRecord files already exist in the directory, then exit without creating the files again
    if _dataset_exists(dataset_dir=FLAGS.dataset_dir, _NUM_SHARDS=FLAGS.number_of_shards):
        print ('Dataset files already exist. Exiting without re-creating them.')
        return None
    # ========================================END OF CHECK===========================================

    # Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories
    photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_dir, FLAGS.folder)

    # Refer each of the class name to a specific integer number for predictions later
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    #Find the number of validation examples we need
    #_NUM_VALIDATION = int(FLAGS.validation_ratio * len(photo_filenames))

    # Divide into train and test:
    random.seed(FLAGS.random_seed)
    random.shuffle(photo_filenames)
    #training_filenames = photo_filenames[_NUM_VALIDATION:]
    #validation_filenames = photo_filenames[:_NUM_VALIDATION]

    # First, convert the training and validation sets.
    _convert_dataset('validation', photo_filenames, class_names_to_ids,
                     dataset_dir=FLAGS.dataset_dir, _NUM_SHARDS = FLAGS.number_of_shards)
    #_convert_dataset('validation', validation_filenames, class_names_to_ids,
                     #dataset_dir)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, FLAGS.dataset_dir)

    #_clean_up_temporary_files(FLAGS.dataset_dir)
    print('\nFinished converting the %s dataset!' %FLAGS.tfrecord_filename)

if __name__ == '__main__':
    main()

