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


def save_tf_record(filename, trajectory_list, agentparams):
    """
    saves data_files from one sample trajectory into one tf-record file
    """
    dir = agentparams['data_save_dir']
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, filename + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    feature = {}

    for tr in range(len(trajectory_list)):

        traj = trajectory_list[tr]

        if 'store_video_prediction' in agentparams:
            sequence_length = len(traj.final_predicted_images)
        else:
            sequence_length = traj.images.shape[0]

        for tind in range(sequence_length):
            feature[str(tind) + '/action']= _float_feature(traj.actions[tind,:].tolist())

            if 'finger_sensors' in agentparams:     #TODO: @Sudeep do you need to change something?
                sdim = agentparams['sdim']
                # using only sdim-1 first values of the state since the gripper position occurs twice.
                arr = np.concatenate([traj.X_full[tind,:sdim-1], traj.touch_sensors[tind]], axis=0)
                assert arr.shape == 7
                feature[str(tind) + '/endeffector_pos'] = _float_feature(arr.tolist())
            else:
                feature[str(tind) + '/endeffector_pos'] = _float_feature(traj.X_Xdot_full[tind,:].tolist())

            if 'cameras' in agentparams:
                for i in range(len(agentparams['cameras'])):
                    image_raw = traj.images[tind, i].tostring()
                    feature[str(tind) + '/image_view{}/encoded'.format(i)] = _bytes_feature(image_raw)
            else:
                if 'store_video_prediction' in agentparams:
                    image_raw = traj.final_predicted_images[tind].tostring()
                else:
                    image_raw = traj.images[tind].tostring()
                feature[str(tind) + '/image_view0/encoded'] = _bytes_feature(image_raw)

            if hasattr(traj, 'Object_pose'):
                Object_pos_flat = traj.Object_pose[tind].flatten()
                feature['move/' + str(tind) + '/object_pos'] = _float_feature(Object_pos_flat.tolist())

                if hasattr(traj, 'max_move_pose'):
                    max_move_pose = traj.max_move_pose[tind].flatten()
                    feature['move/' + str(tind) + '/max_move_pose'] = _float_feature(max_move_pose.tolist())

            if hasattr(traj, 'gen_images'):
                feature[str(tind) + '/gen_images'] = _bytes_feature(traj.gen_images[tind].tostring())
                feature[str(tind) + '/gen_states'] = _float_feature(traj.gen_states[tind,:].tolist())

        if hasattr(traj, 'goal_image'):
            feature['/goal_image'] = _bytes_feature(traj.goal_image.tostring())

        if hasattr(traj, 'first_last_noarm'):
            feature['/first_last_noarm0'] = _bytes_feature(traj.first_last_noarm[0].tostring())
            feature['/first_last_noarm1'] = _bytes_feature(traj.first_last_noarm[1].tostring())

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
    print(('Writing', filename))
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