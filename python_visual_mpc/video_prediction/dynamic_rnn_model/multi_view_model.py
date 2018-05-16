import tensorflow as tf

class Multi_View_Model(object):
    def __init__(self,
                 conf = None,
                 images=None,
                 actions=None,
                 states=None,
                 pix_distrib=None,
                 ):

        if conf['ncam'] == 1:
            self.makenet(0, actions, conf, images, pix_distrib, states)
        else:
            for icam in range(conf['ncam']):
                with tf.variable_scope('icam{}'.format(icam)):
                    self.makenet(icam, actions, conf, images, pix_distrib, states)

    def makenet(self, icam, actions, conf, images, pix_distrib, states):
        images = images[:, :, icam]
        pix_distrib = pix_distrib[:, :, icam]
        Model = conf['pred_model']
        print('using pred_model', Model)
        self.model = Model(conf, images, actions, states, pix_distrib=pix_distrib, build_loss=False)
