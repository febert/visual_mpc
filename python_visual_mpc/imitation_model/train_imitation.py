import tensorflow as tf
from tensorflow.python.platform import flags
import os
import imp
import numpy as np
from python_visual_mpc.video_prediction.read_tf_records2 import \
                    build_tfrecord_input as build_tfrecord
if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_string('pretrained', '', 'pretrained model')
    flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print('using CUDA_VISIBLE_DEVICES=', FLAGS.device)
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    if not os.path.exists(FLAGS.hyper):
        raise RuntimeError("Experiment configuration not found")

    hyperparams = imp.load_source('hyperparams', FLAGS.hyper)
    conf = hyperparams.configuration
    conf['visualize'] = False

    with tf.variable_scope('model', reuse = None) as training_scope:
        data_dict = build_tfrecord(conf, training=True)
        # training input images
        train_images = data_dict['images']
        # training ground truth actions/endef
        train_actions = data_dict['actions']
        train_endeffector_pos = data_dict['endeffector_pos']

        goal_image = data_dict.get('goal_image', None)

        model = conf['model'](conf, train_images, train_actions, train_endeffector_pos, goal_image)
        model.build(is_Train = True)

    with tf.variable_scope('val_model', reuse = None):
        data_dict = build_tfrecord(conf, training=False)
        # validation input images
        val_images = data_dict['images']
        # validation ground truth actions/endef
        val_actions = data_dict['actions']
        val_endeffector_pos = data_dict['endeffector_pos']

        val_goal_image = data_dict.get('goal_image', None)

        with tf.variable_scope(training_scope, reuse=True):
            val_model = conf['model'](conf, val_images, val_actions, val_endeffector_pos, val_goal_image)
            val_model.build(is_Train = False)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    if 'momentum' in conf:
        optimizer = tf.train.MomentumOptimizer(learning_rate, conf['momentum'])
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = 0.95, epsilon=1e-6)
    gradients, variables = zip(*optimizer.compute_gradients(model.loss))
    #gradients = [tf.clip_by_value(g, -conf['clip_grad'], conf['clip_grad']) for g in gradients]
    if 'clip_grad' in conf:
        gradients, _ = tf.clip_by_global_norm(gradients, conf['clip_grad'])
    train_operation = optimizer.apply_gradients(zip(gradients, variables))


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep=0)

    sess = tf.Session(config= tf.ConfigProto(gpu_options=gpu_options))
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())
    
    start_iter = 0
    if len(FLAGS.pretrained) > 0:
        model_name = FLAGS.pretrained
        saver.restore(sess, conf['model_dir'] + model_name)
        start_iter = int(model_name.split('model')[1]) + 1
        print('resuming training at', start_iter)

    summary_writer = tf.summary.FileWriter(conf['model_dir'], graph=sess.graph, flush_secs=10)

    for i in range(start_iter, conf['n_iters']):
        if 'lr_decay' in conf and i > 0 and i % conf['lr_decay'] == 0:
            conf['learning_rate'] /= 5.
        print('iter: {}'.format(i), end='\r')
        
        f_dict = {learning_rate:conf['learning_rate']}
        
        if i % conf['n_print'] == 0:
            sum_line = 'iter: {},\t'.format(i)

            train_model_sums = list(model.summaries.keys())
            val_model_sums = list(val_model.summaries.keys())
            
            all_summaries = [model.summaries[k] for k in train_model_sums]
            all_summaries = all_summaries + [val_model.summaries[k] for k in val_model_sums]
            eval_summaries = sess.run(all_summaries + [train_operation], feed_dict = f_dict)[:-1]
            
            itr_summary = tf.Summary()
            for j, k in enumerate(train_model_sums):
                if k == 'loss':
                    sum_line += 'model loss: {},\t'.format(eval_summaries[j])
                itr_summary.value.add(tag = 'model/{}'.format(k), simple_value = eval_summaries[j])
            for j, k in enumerate(val_model_sums):
                l = j + len(train_model_sums)
                if k == 'loss':
                    sum_line += 'validation loss: {}'.format(eval_summaries[l])
                itr_summary.value.add(tag = 'val_model/{}'.format(k), simple_value = eval_summaries[l])
            summary_writer.add_summary(itr_summary, i)
            print(sum_line)

        else:
            sess.run([train_operation], feed_dict=f_dict)

        if i > 0 and i % conf['n_save'] == 0:
            saver.save(sess, conf['model_dir'] + '/model' + str(i))

    saver.save(sess, conf['model_dir'] + '/modelfinal')
    sess.close()


if __name__ == '__main__':
    main()
