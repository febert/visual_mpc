import tensorflow as tf
from python_visual_mpc.goal_classifier.models.base_model import BaseGoalClassifier
import sys
import os
IS_PYTHON2 = sys.version_info[0] == 2
if IS_PYTHON2:
    import imp
else:
    import importlib.machinery
    import importlib.util


class ClassifierDeploy:
    def __init__(self, conf, checkpoint_path, device_id=0):

        if isinstance(conf, str):
            if IS_PYTHON2:
                params = imp.load_source('params', conf)
                conf = params.config
            else:
                loader = importlib.machinery.SourceFileLoader('classifier_conf', conf)
                spec = importlib.util.spec_from_loader(loader.name, loader)
                mod = importlib.util.module_from_spec(spec)
                loader.exec_module(mod)
                conf = mod.config

        conf.pop('dataset', None)         # if custom data loader is included pop that element
        conf.pop('dataset_params', None)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        g_classifier = tf.Graph()
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True), graph=g_classifier)

        with sess.as_default():
            with g_classifier.as_default():
                model = conf.pop('model', BaseGoalClassifier)(conf)
                with tf.device('/device:GPU:{}'.format(device_id)):
                    model.build()
                model.restore(sess, checkpoint_path)

                def get_scores(**kwargs):
                    kwargs['sess'] = sess    # add session to function arguments
                    return model.score(**kwargs)
                self._score = get_scores

    @property
    def score(self):
        return self._score
