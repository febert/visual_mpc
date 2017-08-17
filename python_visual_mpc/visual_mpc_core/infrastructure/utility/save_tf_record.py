import os
import tensorflow as tf
import numpy as np
import pdb


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def save_tf_record(dir, filename, trajectory_list, params):
    """
    saves data_files from one sample trajectory into one tf-record file
    """

    filename = os.path.join(dir, filename + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    feature = {}

    for tr in range(len(trajectory_list)):

        traj = trajectory_list[tr]

        if 'store_video_prediction' in params:
            sequence_length = len(traj.final_predicted_images)
        else:
            sequence_length = traj._sample_images.shape[0]

        for tind in range(sequence_length):
            if 'store_video_prediction' in params:
                image_raw = traj.final_predicted_images[tind].tostring()
            else:
                image_raw = traj._sample_images[tind].tostring()

            feature['move/' + str(tind) + '/action']= _float_feature(traj.U[tind,:].tolist())
            feature['move/' + str(tind) + '/state'] = _float_feature(traj.X_Xdot_full[tind,:].tolist())
            feature['move/' + str(tind) + '/image/encoded'] = _bytes_feature(image_raw)
            feature['touchdata/' + str(tind)] = _float_feature(traj.touchdata[tind, :].tolist())

            if hasattr(traj, 'Object_pose'):
                Object_pos_flat = traj.Object_pose[tind].flatten()
                feature['move/' + str(tind) + '/object_pos'] = _float_feature(Object_pos_flat.tolist())

                max_move_pose = traj.max_move_pose[tind].flatten()
                feature['move/' + str(tind) + '/max_move_pose'] = _float_feature(max_move_pose.tolist())

            if hasattr(traj, 'large_images_retina'):
                image_raw = traj.large_images_retina[tind].tostring()
                feature['move/' + str(tind) + '/retina/encoded'] = _bytes_feature(image_raw)
                feature['initial_retpos'] = _int64_feature(traj.initial_ret_pos.tolist())

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


def save_tf_record_gtruthpred(dir, filename, trajectory_list, params):
    """
    save both groundtruth and predicted videos from CEM trjaectory
    """

    filename = os.path.join(dir, filename + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    feature = {}

    for tr in range(len(trajectory_list)):
        traj = trajectory_list[tr]
        sequence_length = len(traj.predicted_images)

        for index in range(sequence_length):
            image_raw_pred = traj.predicted_images[index]
            image_raw_pred = (image_raw_pred * 255.).astype(np.uint8).tostring()
            image_raw_gtruth = traj.gtruth_images[index]
            image_raw_gtruth = (image_raw_gtruth).astype(np.uint8).tostring()

            feature['move/' + str(index) + '/image_pred/encoded'] = _bytes_feature(image_raw_pred)
            feature['move/' + str(index) + '/image_gtruth/encoded'] = _bytes_feature(image_raw_gtruth)

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()

def save_tf_record_lval(dir, filename, img_score_list):
    """
    saves data_files from one sample trajectory into one tf-record file
    """

    filename = os.path.join(dir, filename + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    feature = {}

    for ex in range(len(img_score_list)):

        img, score, goalpos, desig_pos, init_state = img_score_list[ex]

        image_raw = img.tostring()

        feature['img'] = _bytes_feature(image_raw)

        feature['score'] = _float_feature([score])
        feature['goalpos'] = _float_feature(goalpos.tolist())
        feature['desig_pos'] = _float_feature(desig_pos.tolist())
        feature['init_state'] = _float_feature(init_state.tolist())

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


    writer.close()