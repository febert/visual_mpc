from .base_dataset import BaseVideoDataset
import tensorflow as tf
import numpy as np


class ClassifierDataset(BaseVideoDataset):
    def _read_manifest(self):
        super(ClassifierDataset, self)._read_manifest()
        num_neg = self._metadata_keys['vidpred_random_actions'][0][1]
        if self._hparams.n_negative <= 0:
            self._hparams.n_negative = num_neg
        elif self._hparams.n_negative > num_neg:
            raise ValueError("N_Negative set to {} but "
                             "dataset only has {} negative examples".format(self._hparams.n_negative, num_neg))
        if self._hparams.n_frames <= 0:
            raise ValueError("n_frames should be >= 1")
        self._index_sel = -tf.random_uniform((), 1, self._hparams.n_frames + 1, dtype=tf.int32)

    def _get_default_hparams(self):
        parent_params = super(ClassifierDataset, self)._get_default_hparams()
        parent_params.add_hparam('cam', 0)
        parent_params.add_hparam('n_negative', 0)  # if this value is greater than 0 only supply up to n_negative bad ex
        parent_params.add_hparam('n_frames', 4)    # select from up to the last n_frames images
        return parent_params

    def _map_key(self, dataset_batch, key):
        if key == 'goal_image':
            goal_images = dataset_batch['ground_truth_video'][:, self._index_sel, self._hparams.cam]
            gathers = []
            for i in range(self._batch_size):
                gathers = gathers + [i for _ in range(self._hparams.n_negative + 1)]
            return tf.gather(goal_images, gathers)
        elif key == 'final_frame':
            real_pred_frames = dataset_batch['vidpred_real_actions'][:, self._index_sel, 0, self._hparams.cam]
            random_pred_frames = dataset_batch['vidpred_random_actions'][:, self._index_sel, :, self._hparams.cam]
            random_pred_frames = random_pred_frames[:, :self._hparams.n_negative]
            frame_batches = []
            for b in range(self._batch_size):
                frame_batches.append(real_pred_frames[b][None])
                frame_batches.append(random_pred_frames[b])
            return tf.concat(frame_batches, axis=0)
        elif key == 'label':
            label_list = np.repeat(np.array([[1] + [0 for _ in range(self._hparams.n_negative)]]), self._batch_size, 0)
            label_list = tf.convert_to_tensor(label_list.reshape(-1), dtype=tf.int32)
            return tf.one_hot(label_list, 2, on_value=1.0, off_value=0.0, axis=-1)
        elif key in dataset_batch:
            return dataset_batch[key]

        raise NotImplementedError('Key {} not present in batch which has keys:\n {}'.format(key,
                                                                                            list(dataset_batch.keys())))