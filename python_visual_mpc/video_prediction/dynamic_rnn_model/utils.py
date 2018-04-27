import tensorflow as tf

import six
import itertools
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import device_setter
from tensorflow.python.util import nest
import numpy as np

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops == None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser


def compute_averaged_gradients(opt, tower_loss, **kwargs):
    tower_gradvars = []
    for loss in tower_loss:
        with tf.device(loss.device):
            gradvars = opt.compute_gradients(loss, **kwargs)
            tower_gradvars.append(gradvars)

    # Now compute global loss and gradients.
    gradvars = []
    with tf.name_scope('gradient_averaging'):
        all_grads = {}
        for grad, var in itertools.chain(*tower_gradvars):
            if grad is not None:
                all_grads.setdefault(var, []).append(grad)
        for var, grads in all_grads.items():
            # Average gradients on the same device as the variables
            # to which they apply.
            with tf.device(var.device):
                if len(grads) == 1:
                    avg_grad = grads[0]
                else:
                    avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
            gradvars.append((avg_grad, var))
    return gradvars

def _reduce_entries(*entries):
    num_gpus = len(entries)
    if entries[0] is None:
        assert all(entry is None for entry in entries[1:])
        reduced_entry = None
    elif isinstance(entries[0], tf.Tensor):
        if entries[0].shape.ndims == 0:
            reduced_entry = tf.add_n(entries) / tf.to_float(num_gpus)
        else:
            reduced_entry = tf.concat(entries, axis=0)
    elif np.isscalar(entries[0]) or isinstance(entries[0], np.ndarray):
        if np.isscalar(entries[0]) or entries[0].ndim == 0:
            reduced_entry = sum(entries) / float(num_gpus)
        else:
            reduced_entry = np.concatenate(entries, axis=0)
    elif isinstance(entries[0], tuple) and len(entries[0]) == 2:
        losses, weights = zip(*entries)
        loss = tf.add_n(losses) / tf.to_float(num_gpus)
        if isinstance(weights[0], tf.Tensor):
            with tf.control_dependencies([tf.assert_equal(weight, weights[0]) for weight in weights[1:]]):
                weight = tf.identity(weights[0])
        else:
            assert all(weight == weights[0] for weight in weights[1:])
            weight = weights[0]
        reduced_entry = (loss, weight)
    else:
        raise NotImplementedError
    return reduced_entry


def reduce_tensors(structures, shallow=False):
    if len(structures) == 1:
        reduced_structure = structures[0]
    else:
        if shallow:
            if isinstance(structures[0], dict):
                shallow_tree = type(structures[0])([(k, None) for k in structures[0]])
            else:
                shallow_tree = type(structures[0])([None for _ in structures[0]])
            reduced_structure = nest.map_structure_up_to(shallow_tree, _reduce_entries, *structures)
        else:
            reduced_structure = nest.map_structure(_reduce_entries, *structures)
    return reduced_structure


def _as_name_scope_map(values):
    name_scope_to_values = {}
    for name, value in values.items():
        name_scope = "%s_summary" % name.split('/')[0]
        name_scope_to_values.setdefault(name_scope, {})
        name_scope_to_values[name_scope][name] = value
    return name_scope_to_values

def add_scalar_summaries(losses_or_metrics, collections=None):
    for name_scope, losses_or_metrics in _as_name_scope_map(losses_or_metrics).items():
        with tf.name_scope(name_scope):
            for name, loss_or_metric in losses_or_metrics.items():
                if isinstance(loss_or_metric, tuple):
                    loss_or_metric, _ = loss_or_metric
                tf.summary.scalar(name, loss_or_metric, collections=collections)


def transpose_batch_time(x):
    if isinstance(x, tf.Tensor) and x.shape.ndims >= 2:
        return tf.transpose(x, [1, 0] + list(range(2, x.shape.ndims)))
    else:
        return x


def cuttoff_gen_tsteps(x, icutoff):
    """
    :param x:
    :param icutoff: cut everything off that is before this index in the first dimension
    :return:
    """
    if isinstance(x, tf.Tensor) and x.shape.ndims >= 2:
        return x[icutoff:]
    else:
        return x
