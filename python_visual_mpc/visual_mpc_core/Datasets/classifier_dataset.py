from .base_dataset import BaseVideoDataset
import tensorflow as tf


class ClassifierDataset(BaseVideoDataset):
    def _read_manifest(self):
        super(ClassifierDataset, self)._read_manifest()
        num_neg = self._metadata_keys['vidpred_random_actions'][0][1]
        if self._hparams.n_negative <= 0:
            self._hparams.n_negative = num_neg
        elif self._hparams.n_negative > num_neg:
            raise ValueError("N_Negative set to {} but "
                             "dataset only has {} negative examples".format(self._hparams.n_negative, num_neg))

    def _get_default_hparams(self):
        parent_params = super(ClassifierDataset, self)._get_default_hparams()
        parent_params.add_hparam('cam', 0)
        parent_params.add_hparam('n_negative', 0)  # if this value is greater than 0 only supply up to n_negative bad ex
        return parent_params

    def _map_key(self, dataset_batch, key):
        if key == 'goal_image':
            return dataset_batch['ground_truth_video'][:, -1, self._hparams.cam]
        elif key == 'final_frame':

            # vidpred_real_actions vidpred_random_actions
            return None
        elif key in dataset_batch:
            return dataset_batch[key]

        raise NotImplementedError('Key {} not present in batch which has keys:\n {}'.format(key,
                                                                                            list(dataset_batch.keys())))