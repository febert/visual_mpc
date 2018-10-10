
from python_visual_mpc.embedding_metric.utils.load_data import load_benchmark_data
import numpy as np
import tensorflow as tf



def build_model(input_sequences):
    """
    :param input_sequences: shape: bsize, t, r, c, 3
    :return:
    """

    #make the variable:

    bsize, seqlen, height, width, _ = input_sequences.shape





    low_dist_pairs = tf.random_uniform([])


    high_dist_pairs


    n_layer = 3
    h_0 = 32

    weight = []
    loss = 0.

    for n in range(n_layer):
        weight.append(tf.get_variable("w", [h_0/(n + 1), h_0/(n + 1), 8*2**n], dtype=tf.int32,
                        initializer=tf.zeros_initializer))







def train_weighting():

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.summary.FileWriter(conf['event_log_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())


    for itr in range(1e3):

        feed_dict = {model.iter_num: np.float32(itr),
                     model.train_cond: 1}

        cost, _, summary_str, lr = sess.run([model.loss, model.train_op, model.train_summ_op, model.learning_rate],
                                            feed_dict)

        if (itr) % 100 == 0:
            tf.logging.info(str(itr) + ' ' + str(cost))
            tf.logging.info('lr: {}'.format(lr))

        if (itr) % 100 == 2:
            # Run through validation set.
            feed_dict = {model.iter_num: np.float32(itr),
                         model.train_cond: 0}
            [val_summary_str] = sess.run([model.val_summ_op], feed_dict)
            summary_writer.add_summary(val_summary_str, itr)

        if (itr) % 100 == 0:
            summary_writer.add_summary(summary_str, itr)