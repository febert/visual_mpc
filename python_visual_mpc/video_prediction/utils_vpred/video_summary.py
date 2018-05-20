import tempfile
import moviepy.editor as mpy
import numpy as np
import tensorflow as tf


def make_video_summaries(ncntxt, videos, name):
    if not isinstance(videos[0], list):
        videos = [tf.unstack(v, axis=1) for v in videos]
    seq_len = len(videos[0])
    columns = []
    videos = [vid[-(seq_len-ncntxt):] for vid in videos]
    for t in range(seq_len-ncntxt):
        colimages = [vid[t] for vid in videos]
        columns.append(tf.concat(colimages, axis=1))
    summary = tf.summary.tensor_summary(name , tf.cast(tf.stack(columns, axis=1) * 255, tf.uint8))
    return summary

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