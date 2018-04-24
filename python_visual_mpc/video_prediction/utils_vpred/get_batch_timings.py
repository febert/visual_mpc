import sys
import pickle
import imp
import os
from tensorflow.python.platform import flags

from python_visual_mpc.video_prediction.trainvid import main
import numpy as np


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_string('visualize_check', "", 'model within hyperparameter folder from which to create gifs')
    flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
    flags.DEFINE_string('resume', None, 'path to model file from which to resume training')
    flags.DEFINE_bool('diffmotions', False, 'visualize several different motions for a single scene')
    flags.DEFINE_bool('metric', False, 'compute metric of expected distance to human-labled positions ob objects')
    flags.DEFINE_bool('float16', False, 'whether to do inference with float16')
    flags.DEFINE_bool('create_images', False, 'whether to create images')
    flags.DEFINE_bool('ow', False, 'overwrite previous experiment')

batch_size = [1,2,3,4,8,16,32]

average_times = []
for bsize in batch_size:
    conf_file = FLAGS.hyper

    if not os.path.exists(FLAGS.hyper):
        sys.exit("Experiment configuration not found")
    hyperparams = imp.load_source('hyperparams', conf_file)

    conf = hyperparams.configuration

    conf['batch_size'] = bsize
    conf['timingbreak'] = 200
    itertimes = main(None, conf, FLAGS)

    # taking the average over the last fifty to avoid measureing slower warmup phase
    average_times.append(np.mean(itertimes[-50:]))
    print('##################################')
    print('average iteration time with batchsize {}: {}'.format(bsize, average_times[-1]))

pickle.dump({'average_times':average_times, 'batch_sizes':batch_size} , open('average_times.pkl','wb'))
for i in range(len(batch_size)):
    print('batch size {} average time {}'.format(batch_size[i], average_times[i]))

