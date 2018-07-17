import os

from tensorflow.contrib.training import HParams
from tensorflow.python.util import nest
import pdb

from.utils import tf_utils


class BaseVideoPredictionModel(object):
    def __init__(self, mode='train', hparams_dict=None, hparams=None,
                 num_gpus=None, eval_num_samples=100, eval_parallel_iterations=1):
        """
        Base video prediction model.

        Trainable and non-trainable video prediction models can be derived
        from this base class.

        Args:
            hparams_dict: a dict of `name=value` pairs, where `name` must be
                defined in `self.get_default_hparams()`.
            hparams: a string of comma separated list of `name=value` pairs,
                where `name` must be defined in `self.get_default_hparams()`.
                These values overrides any values in hparams_dict (if any).
        """
        self.mode = mode
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        if cuda_visible_devices == '':
            max_num_gpus = 0
        else:
            max_num_gpus = len(cuda_visible_devices.split(','))
        if num_gpus is None:
            num_gpus = max_num_gpus
        elif num_gpus > max_num_gpus:
            raise ValueError('num_gpus=%d is greater than the number of visible devices %d' % (num_gpus, max_num_gpus))
        self.num_gpus = num_gpus
        self.eval_num_samples = eval_num_samples
        self.eval_parallel_iterations = eval_parallel_iterations
        self.hparams = self.parse_hparams(hparams_dict, hparams)

        # should be overriden by descendant class if the model is stochastic
        self.deterministic = True

        # member variables that should be set by `self.build_graph`
        self.inputs = None

    def get_default_hparams_dict(self):
        """
        The keys of this dict define valid hyperparameters for instances of
        this class. A class inheriting from this one should override this
        method if it has a different set of hyperparameters.

        Returns:
            A dict with the following hyperparameters.
            ?
        """
        hparams = dict(
            test=1
        )
        return hparams

    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())

    def parse_hparams(self, hparams_dict, hparams):
        parsed_hparams = self.get_default_hparams().override_from_dict(hparams_dict or {})
        if hparams:
            if not isinstance(hparams, (list, tuple)):
                hparams = [hparams]
            for hparam in hparams:
                parsed_hparams.parse(hparam)
        return parsed_hparams

    def build_graph(self, inputs, targets=None):
        self.inputs = inputs
        self.targets = targets

    def restore(self, sess, checkpoints):

        if checkpoints:
            # possibly restore from multiple checkpoints. useful if subset of weights
            # (e.g. generator or discriminator) are on different checkpoints.
            if not isinstance(checkpoints, (list, tuple)):
                checkpoints = [checkpoints]
            # automatically skip global_step if more than one checkpoint is provided
            skip_global_step = len(checkpoints) > 1
            savers = []
            for checkpoint in checkpoints:
                print("creating restore saver from checkpoint %s" % checkpoint)
                saver, _ = tf_utils.get_checkpoint_restore_saver(checkpoint, skip_global_step=skip_global_step)
                savers.append(saver)
            restore_op = [saver.saver_def.restore_op_name for saver in savers]
            sess.run(restore_op)
