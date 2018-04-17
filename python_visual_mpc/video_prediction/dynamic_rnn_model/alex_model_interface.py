import tensorflow as tf
from video_prediction.models import SAVPVideoPredictionModel
import json
import os

class Alex_Interface_Model(object):
    def __init__(self,
                 conf = None,
                 images=None,
                 actions=None,
                 states=None,
                 pix_distrib= None,
                 build_loss = False
                 ):

        with open(os.path.join(conf['json_dir'], "model_hparams.json")) as f:
            model_hparams_dict = json.loads(f.read())
            model_hparams_dict.pop('num_gpus', None)  # backwards-compatibility

        self.m = SAVPVideoPredictionModel(mode='test', hparams_dict=model_hparams_dict)

        inputs = {
            'images':images,
            'actions':actions,
            'states':states
        }
        if pix_distrib is not None:
            inputs['pix_distribs'] = pix_distrib
        self.m.build_graph(inputs)

        self.gen_images = tf.unstack(self.m.outputs['gen_images'], axis=1)
        self.gen_states = tf.unstack(self.m.outputs['gen_states'], axis=1)

        if pix_distrib is not None:
            self.gen_distrib = tf.unstack(self.m.outputs['gen_pix_distribs'], axis=1)
