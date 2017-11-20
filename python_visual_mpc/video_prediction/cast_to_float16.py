import tensorflow as tf
from tensorflow.python.platform import flags
import imp
from prediction_train_sawyer import Model, filter_vars

"""

"""
def main(conf, model):
    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test'])
    conf['visualize'] = conf['output_dir'] + '/' + model
    conf['event_log_dir'] = '/tmp'
    conf.pop('use_len', None)
    conf['batch_size'] = 32

    conf['sequence_length'] = 15

    if 'sawyer' in conf:
        from read_tf_record_sawyer12 import build_tfrecord_input
    else:
        from read_tf_record import build_tfrecord_input

    with tf.variable_scope('half_float', reuse=None):
        val_images_aux1, val_actions, val_states = build_tfrecord_input(conf, training=False)
        val_images = val_images_aux1

        val_images = tf.cast(val_images, tf.float16)
        val_actions = tf.cast(val_actions, tf.float16)
        val_states = tf.cast(val_states, tf.float16)

        half_float = Model(conf, val_images, val_actions, val_states, inference=False)

    with tf.variable_scope('model', reuse=None) as training_scope:
        images_aux1, actions, states = build_tfrecord_input(conf, training=True)
        images = images_aux1

        model = Model(conf, images, actions, states, inference=False)

    with tf.variable_scope('val_model', reuse=None):
        val_images_aux1, val_actions, val_states = build_tfrecord_input(conf, training=False)
        val_images = val_images_aux1

        val_model = Model(conf, val_images, val_actions, val_states,
                          training_scope, inference=False)



    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model') + tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='val_model')
    vars_no_state = filter_vars(vars)
    model_saver = tf.train.Saver(vars_no_state, max_to_keep=0)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='half_float')
    vars_no_state = filter_vars(vars)
    half_model_saver = tf.train.Saver(vars_no_state, max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(tf.global_variables_initializer())
    print 'Loading model', conf['visualize']
    model_saver.restore(sess, conf['visualize'])

    print 'sucessfully restored!'

    print 'beginning casting'
    for i, j in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='half_float'),
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')):
        print 'casting', j.name, 'to', i.name
        sess.run(tf.assign(i, tf.cast(j, i.dtype)))

    print 'casted!'
    half_model_saver.save(sess, conf['visualize'] + 'float16')
    print 'Save casted to', conf['visualize'] + 'float16'

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_string('model', '', 'model within hyperparameter folder to cast to float16')

    hyperparams = imp.load_source('hyperparams', FLAGS.hyper)

    conf = hyperparams.configuration
    main(conf, FLAGS.model)