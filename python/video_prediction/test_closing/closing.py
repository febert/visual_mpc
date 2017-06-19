from video_prediction.transformer.spatial_transformer import transformer

import matplotlib.pyplot as plt

import tensorflow as tf

from PIL import Image
import numpy as np


input_image = np.zeros((64, 64, 1),dtype= np.float32)
input_image[24:32,24:32] = 1
input_image[32:41,32:41] = 1

# Image.fromarray((np.squeeze(input_image)*255).astype(np.uint8)).show()
fig = plt.figure()
plt.imshow(np.squeeze(input_image), interpolation='None')
plt.show()

input_image = input_image.reshape([1,64,64,1], )

input_pl = tf.placeholder(tf.float32, [1,64,64,1])




# Make training session.
sess = tf.InteractiveSession()

tf.train.start_queue_runners(sess)
sess.run(tf.initialize_all_variables())

feed_dict = {
    input_pl: input_image,
}

transformed_data = sess.run([transformed],feed_dict)


Image.fromarray((np.squeeze(transformed_data)*255.).astype(np.uint8)).show()





