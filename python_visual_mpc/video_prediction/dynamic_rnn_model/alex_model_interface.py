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

        self.gen_images = tf.unstack(self.model.gen_images, axis=1)
        # self.gen_masks = tf.unstack(self.model.gen_masks)
        self.prediction_flow = tf.unstack(self.model.gen_flow_map, axis=1)
        if pix_distrib is not None:
            self.gen_distrib = tf.unstack(self.model.gen_pix_distribs, axis=1)
        self.gen_states = tf.unstack(self.model.gen_states)
        self.gen_masks = self.model.gen_masks

        with open(os.path.join(conf['checkpoint_dir'], "model_hparams.json")) as f:
            model_hparams_dict = json.loads(f.read())
            model_hparams_dict.pop('num_gpus', None)  # backwards-compatibility

        self.model = SAVPVideoPredictionModel(mode='test', hparams_dict=model_hparams_dict)

        inputs = {
            'images':images,
            'actions':actions,
            'states':states
        }
        if pix_distrib is not None:
            inputs['pix_distribs'] = pix_distrib
        self.model.build_graph(inputs)

        self.gen_images = tf.unstack(self.model.outputs['gen_images'], axis=1)
        self.gen_states = tf.unstack(self.model.outputs['gen_states'], axis=1)

        if pix_distrib is not None:
            self.gen_pixdistrib = tf.unstack(self.model.outputs['gen_pix_distrib'], axis=1)
