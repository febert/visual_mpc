import tensorflow as tf
import numpy as np


# Features are length-100 vectors of floats
feature_input = tf.placeholder(tf.float32, shape=[100])
# Labels are scalar integers.
label_input = tf.placeholder(tf.int32, shape=[])

# Alternatively, could do:
# feature_batch_input = tf.placeholder(tf.float32, shape=[None, 100])
# label_batch_input = tf.placeholder(tf.int32, shape=[None])

q = tf.FIFOQueue(100, [tf.float32, tf.int32], shapes=[[100], []])
enqueue_op = q.enqueue([feature_input, label_input])

# For batch input, do:
# enqueue_op = q.enqueue_many([feature_batch_input, label_batch_input])
BATCH_SIZE = 10
feature_batch, label_batch = q.dequeue_many(BATCH_SIZE)
# Build rest of model taking label_batch, feature_batch as input.
# [...]

sess = tf.Session()


def load_and_enqueue():
  with open(...) as feature_file, open(...) as label_file:
    while True:
      feature_array = numpy.fromfile(feature_file, numpy.float32, 100)
      if not feature_array:
        return
      label_value = numpy.fromfile(feature_file, numpy.int32, 1)[0]

      sess.run(enqueue_op, feed_dict={feature_input: feature_array,
                                      label_input: label_value})

# Start a thread to enqueue data asynchronously, and hide I/O latency.
t = threading.Thread(target=load_and_enqueue)
t.start()