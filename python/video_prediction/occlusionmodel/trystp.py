from video_prediction.transformer.spatial_transformer import transformer


import tensorflow as tf

from PIL import Image
import numpy as np


input_image = np.zeros((64, 64, 1),dtype= np.float32)
input_image[30:34,:] = 1
input_image[:, 30:34] = 1

Image.fromarray((np.squeeze(input_image)*255).astype(np.uint8)).show()
input_image = input_image.reshape([1,64,64,1])


identity_params = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]], np.float32)

trafo = identity_params
shift_vec = np.array([0.5, 0.5])
trafo[:,2] += shift_vec
trafo = trafo.reshape([1,6])

input_pl = tf.placeholder(tf.float32, [1,64,64,1])
trafo_pl = tf.placeholder(tf.float32, [1,6])

transformed = transformer(input_pl, trafo_pl, np.array([64, 64]))


# Make training session.
sess = tf.InteractiveSession()

tf.train.start_queue_runners(sess)
sess.run(tf.initialize_all_variables())

feed_dict = {
    input_pl: input_image,
    trafo_pl: trafo
}

transformed_data = sess.run([transformed],feed_dict)


Image.fromarray((np.squeeze(transformed_data)*255.).astype(np.uint8)).show()





