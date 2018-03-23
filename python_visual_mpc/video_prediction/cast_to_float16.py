import tensorflow as tf
from tensorflow.python.platform import flags
import imp
from .prediction_train_sawyer import Model, filter_vars
from python_visual_mpc.video_prediction.utils_vpred.variable_checkpoint_matcher import variable_checkpoint_matcher
"""

"""
def main(conf, model):
    conf['schedsamp_k'] = -1  # don't feed ground truth
    # conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test'])
    source_model_file = conf['output_dir'] + '/' + model
    conf['event_log_dir'] = '/tmp'
    conf.pop('use_len', None)
    conf['batch_size'] = 32



    conf.pop('visual_flowvec', None)

    conf['sequence_length'] = 4 #####

    Model = conf['pred_model']
    model = Model(conf, load_data=False, trafo_pix=False, build_loss=False)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    orig_vars = vars = filter_vars(vars)
    vars = variable_checkpoint_matcher(conf, vars, source_model_file)
    orig_model_saver = tf.train.Saver(vars, max_to_keep=0)

    with tf.variable_scope('half_float', reuse=None):
        conf['float16'] = ''
        images_pl = tf.placeholder(tf.float16, name='images',
                                   shape=(
                                   conf['batch_size'], conf['sequence_length'], conf['img_height'], conf['img_width'],
                                   3))
        sdim = conf['sdim']
        adim = conf['adim']
        print('adim', adim)
        print('sdim', sdim)
        actions_pl = tf.placeholder(tf.float16, name='actions',
                                    shape=(conf['batch_size'], conf['sequence_length'], adim))
        states_pl = tf.placeholder(tf.float16, name='states',
                                   shape=(conf['batch_size'], conf['sequence_length'], sdim))

        half_float = Model(conf, images = images_pl, actions = actions_pl, states= states_pl,
                           load_data=True, trafo_pix=False, build_loss=False)

    half_float_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='half_float')
    half_float_vars = filter_vars(half_float_vars)
    half_model_saver = tf.train.Saver(half_float_vars, max_to_keep=0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(tf.global_variables_initializer())
    print('Loading model', source_model_file)
    orig_model_saver.restore(sess, source_model_file)

    print('sucessfully restored!')

    print('beginning casting')
    for hf in half_float_vars:
        for orig in orig_vars:
            if '/'.join(str.split(str(hf.name), '/')[1:]) == str(orig.name):
                print('casting', orig.name, 'to', hf.name)
                sess.run(tf.assign(hf, tf.cast(orig, hf.dtype)))

    print('casted!')
    half_model_saver.save(sess, source_model_file + 'float16')
    print('Save casted to', source_model_file + 'float16')

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_string('model', '', 'model within hyperparameter folder to cast to float16')

    hyperparams = imp.load_source('hyperparams', FLAGS.hyper)

    conf = hyperparams.configuration
    main(conf, FLAGS.model)