import tensorflow as tf
import numpy as np
from python_visual_mpc.visual_mpc_core.Datasets.base_dataset import BaseVideoDataset


class InferLabelDataset(BaseVideoDataset):
    def __init__(self, directory, batch_size, hparams_dict=dict()):
        super(InferLabelDataset, self).__init__(directory, batch_size, hparams_dict)
        if 'positive' in directory:
            label_list = np.ones(batch_size, dtype=np.int32)
        elif 'negative' in directory:
            label_list = np.zeros(batch_size, dtype=np.int32)
        else:
            raise ValueError("can't infer label of: {}".format(directory))
        label_list = tf.convert_to_tensor(label_list.reshape(-1), dtype=tf.int32)
        self._label = tf.one_hot(label_list, 2, on_value=1.0, off_value=0.0, axis=-1)

    def _map_key(self, dataset_batch, key):
        if key == 'label':
            return self._label
        elif key == 'images':
            images = super(InferLabelDataset, self)._map_key(dataset_batch, key)
            return images[:, 0]
        raise NotImplementedError("Does not have key: {}".format(key))
