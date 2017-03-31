import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def test_fft():


    img = Image.open('test.png')
    img.show()
    img = np.asarray(img)


    input_image = tf.placeholder(tf.float32, shape= [64, 64, 3])
    img = img.astype(np.float32) / 255.

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        feed_dict = {input_image: img,
                     }

        fft_abs = 0

        for i in range(3):
            input_slice = tf.slice(input_image, [0, 0, i], [-1, -1, 1])
            input_slice = tf.squeeze(tf.complex(input_slice, tf.zeros_like(input_slice)))
            fft = tf.fft2d(input_slice)
            fft_abs += tf.complex_abs(fft)

        fft_res = sess.run([fft_abs], feed_dict= feed_dict)
        fft_res = np.clip(fft_res, 0, 10)
        fft_res = np.squeeze(fft_res)

        # fft_res = fft_res[5:59, 5:59]
        plt.imshow(fft_res)
        plt.colorbar()
        plt.show()


        res_img = Image.fromarray((fft_res*255.).astype(np.uint8))
        res_img.show()


if __name__ == '__main__':
    test_fft()