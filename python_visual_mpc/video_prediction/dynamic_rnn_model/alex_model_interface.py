import tensorflow as tf
from video_prediction.models import MultiSAVPVideoPredictionModel
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
            if 'override_json' in conf:
                model_hparams_dict.update(conf['override_json'])

        self.m = MultiSAVPVideoPredictionModel(mode='test', hparams_dict=model_hparams_dict)

        images, images1 = tf.unstack(images, conf['ncam'], 2)
        inputs = {
            'images':images,
            'images1':images1,
            'actions':actions,
            'states':states
        }
        if pix_distrib is not None: # input batch , t, ncam, r, c, ndesig
            inputs['pix_distribs']  = pix_distrib[:,:,0]
            inputs['pix_distribs1'] = pix_distrib[:,:,1]
        self.m.build_graph(inputs)

        self.gen_images = tf.stack([self.m.outputs['gen_images'],   #ouput  b, t, ncam, r, c, 3
                                     self.m.outputs['gen_images1']], axis=2)
        self.gen_states = self.m.outputs['gen_states']

        if pix_distrib is not None:
            self.gen_distrib = tf.stack([self.m.outputs['gen_pix_distribs'],
                                         self.m.outputs['gen_pix_distribs1']], axis=2)
