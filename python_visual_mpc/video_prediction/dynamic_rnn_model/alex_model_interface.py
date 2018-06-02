import tensorflow as tf
from video_prediction.models import MultiSAVPVideoPredictionModel, SAVPVideoPredictionModel
import json
import os
import pdb

class Alex_Interface_Model(object):
    def __init__(self,
                 conf = None,
                 images=None,
                 actions=None,
                 states=None,
                 pix_distrib= None,
                 build_loss = False,
                 load_data = True
                 ):


        with open(os.path.join(conf['json_dir'], "model_hparams.json")) as f:
            model_hparams_dict = json.loads(f.read())
            model_hparams_dict.pop('num_gpus', None)  # backwards-compatibility
            if 'override_json' in conf:
                model_hparams_dict.update(conf['override_json'])

        with open(os.path.join(conf['json_dir'], "dataset_hparams.json")) as f:
            datatset_hparams_dict = json.loads(f.read())

        self.adim = datatset_hparams_dict.adim
        if hasattr(datatset_hparams_dict, 'auto_grasp'):
            self.adim = datatset_hparams_dict.auto_grasp
        self.sdim = datatset_hparams_dict.sdim
        ndesig = conf['ndesig']

        self.img_height, self.img_width = conf['orig_size']

        seq_len = model_hparams_dict.sequence_length
        nctxt = model_hparams_dict.context_frames
        if images is None:
            self.actions_pl = tf.placeholder(tf.float32, name='actions',
                                             shape=(conf['batch_size'], seq_len, self.adim))
            actions = self.actions_pl
            self.states_pl = tf.placeholder(tf.float32, name='states',
                                            shape=(conf['batch_size'], seq_len, self.sdim))
            states = self.states_pl
            self.images_pl = tf.placeholder(tf.float32, name='images',
                                            shape=(conf['batch_size'], seq_len, self.img_height, self.img_width, 3))
            images = self.images_pl
            self.pix_distrib_pl = tf.placeholder(tf.float32, name='states',
                                                 shape=(conf['batch_size'], seq_len, ndesig, self.img_height, self.img_width, 1))
            pix_distrib = self.pix_distrib_pl

            targets = {'images':images[nctxt:], 'states':states[nctxt:]}
        else:
            targets = None

        if conf['ncam'] == 1:
            self.m = SAVPVideoPredictionModel(mode='test', hparams_dict=model_hparams_dict)
        elif conf['ncam'] == 2:
            self.m = MultiSAVPVideoPredictionModel(mode='test', hparams_dict=model_hparams_dict)

        inputs = {'actions':actions, 'states':states,'images':images[:,:,0]}
        if conf['ncam'] == 2:
            inputs['images1'] = images[:,:,1]

        if pix_distrib is not None: # input batch , t, ncam, r, c, ndesig
            inputs['pix_distribs'] = pix_distrib[:,:,0]
            if conf['ncam'] == 2:
                inputs['pix_distribs1'] = pix_distrib[:,:,1]

        self.m.build_graph(inputs, targets)

        gen_images = [self.m.outputs['gen_images']]
        if conf['ncam'] == 2:
            gen_images.append(self.m.outputs['gen_images1'])
        self.gen_images = tf.stack(gen_images, axis=2) #ouput  b, t, ncam, r, c, 3
        self.gen_states = self.m.outputs['gen_states']

        if pix_distrib is not None:
            gen_distrib = [self.m.outputs['gen_pix_distribs']]
            if conf['ncam'] == 2:
                gen_distrib.append(self.m.outputs['gen_pix_distribs1'])
            self.gen_distrib = tf.stack(gen_distrib, axis=2) #ouput  b, t, ncam, r, c, 3
