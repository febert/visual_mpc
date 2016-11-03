import tensorflow as tf
import numpy as np

x = np.arange(9*3)
x = x.reshape((3,3,3))
x = tf.Variable(x)
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

print(x.eval())

y = tf.gather_nd(x,[[0, 0], [1, 1], [0, 0], [2, 2]])

print y.eval()


# indices = np.array([])
# indices = np.arange(15)*2
# indices = np.tile(indices, (32,1))
#
# print a.get_shape()