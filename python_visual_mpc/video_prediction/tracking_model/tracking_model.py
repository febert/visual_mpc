import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from python_visual_mpc.video_prediction.lstm_ops12 import basic_conv_lstm_cell
from python_visual_mpc.misc.zip_equal import zip_equal

import pdb

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

from python_visual_mpc.video_prediction.basecls.prediction_model_basecls import Base_Prediction_Model

class Tracking_Model(Base_Prediction_Model):
    def __init__(self, images,
                        actions=None,
                        states=None,
                        iter_num=-1.0,
                        pix_distrib1=None,
                        pix_distrib2=None,
                        conf = None):
            Base_Prediction_Model.__init__( images,
                                            actions=None,
                                            states=None,
                                            iter_num=-1.0,
                                            pix_distrib1=None,
                                            pix_distrib2=None,
                                            conf = None)


    def build_tracker(self):

        self.
