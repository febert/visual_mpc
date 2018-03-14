import tensorflow as tf
from tensorflow.python.platform import flags
import os
import imp


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
    flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print 'using CUDA_VISIBLE_DEVICES=', FLAGS.device
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()

    if not os.path.exists(FLAGS.hyper):
        raise RuntimeError("Experiment configuration not found")

    hyperparams = imp.load_source('hyperparams', FLAGS.hyper)
    conf = hyperparams.configuration
    conf['visualize'] = False





if __name__ == '__main__':
    main()