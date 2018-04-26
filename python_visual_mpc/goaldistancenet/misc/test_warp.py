import tensorflow as tf
from PIL import  Image
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def get_coords(img_shape):
    y = tf.cast(tf.range(img_shape[1]), tf.float32)
    x = tf.cast(tf.range(img_shape[2]), tf.float32)
    batch_size = img_shape[0]

    X,Y = tf.meshgrid(x,y)
    coords = tf.expand_dims(tf.stack((X, Y), axis=2), axis=0)
    coords = tf.tile(coords, [batch_size, 1,1,1])
    return coords

def resample_layer(src_img, warp_pts, name="tgt_img"):
    with tf.variable_scope(name):
        return tf.contrib.resampler.resampler(src_img, warp_pts)

def warp_pts_layer(flow_field, name="warp_pts"):
    with tf.variable_scope(name):
        img_shape = flow_field.get_shape().as_list()
        return flow_field + get_coords(img_shape)


im = np.array(Image.open('im0.png')).astype(np.float32)/255.
im = im[15:63]

im = im.reshape([1,48,64,3])

flow_field = np.zeros([1, im.shape[1], im.shape[2], 2])

im_pl = tf.placeholder(tf.float32, im.shape)
flow_field_pl = tf.placeholder(tf.float32, flow_field.shape)



warp_pts = warp_pts_layer(flow_field_pl)
warped = resample_layer(im, warp_pts)


sess = tf.InteractiveSession()

[warped_im, warp_pts] = sess.run([warped, warp_pts], feed_dict={flow_field_pl: flow_field, im_pl: im})

plt.imshow(warped_im)
plt.show()

