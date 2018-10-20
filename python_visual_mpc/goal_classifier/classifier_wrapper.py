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
    def __init__(self, conf_path, checkpoint_path, device_id=0):
        if device_id == None:
            device_id = 0

        os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
        print('using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"])

        if IS_PYTHON2:
            params = imp.load_source('params', conf_path)
            conf = params.config
        else:
            loader = importlib.machinery.SourceFileLoader('classifier_conf', conf_path)
            spec = importlib.util.spec_from_loader(loader.name, loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            conf = mod.config

        conf.pop('dataset', None)         # if custom data loader is included pop that element

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        g_classifier = tf.Graph()
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g_classifier)

        with sess.as_default():
            with g_classifier.as_default():
                model = conf.pop('model', BaseGoalClassifier)(conf)
                model.build()
                model.restore(sess, checkpoint_path)

                def get_scores(**kwargs):
                    kwargs['sess'] = sess    # add session to function arguments
                    return model.score(**kwargs)
                self._score = get_scores

    @property
    def score(self):
        return self._score
