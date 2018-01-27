import tempfile
import moviepy.editor as mpy
import numpy as np
import tensorflow as tf

def convert_tensor_to_gif_summary(summ):
    if isinstance(summ, bytes):
        summary_proto = tf.Summary()
        summary_proto.ParseFromString(summ)
        summ = summary_proto

    summary = tf.Summary()
    for value in summ.value:
        tag = value.tag
        images_arr = tf.make_ndarray(value.tensor)

        if len(images_arr.shape) == 5:
            # concatenate batch dimension horizontally
            images_arr = np.concatenate(list(images_arr), axis=-2)
        if len(images_arr.shape) != 4:
            raise ValueError('Tensors must be 4-D or 5-D for gif summary.')
        if images_arr.shape[-1] != 3:
            raise ValueError('Tensors must have 3 channels.')

        # encode sequence of images into gif string
        clip = mpy.ImageSequenceClip(list(images_arr), fps=4)
        with tempfile.NamedTemporaryFile() as f:
            filename = f.name + '.gif'
        clip.write_gif(filename, verbose=False)
        with open(filename, 'rb') as f:
            encoded_image_string = f.read()

        image = tf.Summary.Image()
        image.height = images_arr.shape[-3]
        image.width = images_arr.shape[-2]
        image.colorspace = 3  # code for 'RGB'
        image.encoded_image_string = encoded_image_string
        summary.value.add(tag=tag, image=image)
    return summary

sess = tf.Session()
summary_writer = tf.summary.FileWriter('logs/image_summary', graph=tf.get_default_graph())

images_shape = (16, 12, 64, 64, 3)  # batch, time, height, width, channels
images = np.random.randint(256, size=images_shape).astype(np.uint8)
images = tf.convert_to_tensor(images)

tensor_summ = tf.summary.tensor_summary('images_gif', images)
tensor_value = sess.run(tensor_summ)
summary_writer.add_summary(convert_tensor_to_gif_summary(tensor_value), 0)

summ = tf.summary.image("images", images[:, 0])  # first time-step only
value = sess.run(summ)
summary_writer.add_summary(value, 0)

summary_writer.flush()