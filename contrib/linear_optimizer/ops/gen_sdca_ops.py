"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import op_def_library


_distributed_sdca_large_batch_solver_outputs = ["out_example_state_data",
                                               "out_delta_sparse_weights",
                                               "out_delta_dense_weights"]


_DistributedSdcaLargeBatchSolverOutput = collections.namedtuple("DistributedSdcaLargeBatchSolver",
                                                                _distributed_sdca_large_batch_solver_outputs)


def distributed_sdca_large_batch_solver(sparse_example_indices,
                                        sparse_feature_indices,
                                        sparse_feature_values, dense_features,
                                        example_weights, example_labels,
                                        sparse_weights, dense_weights,
                                        example_state_data, loss_type, l1, l2,
                                        num_loss_partitions,
                                        num_inner_iterations, adaptative=None,
                                        name=None):
  r"""Distributed version of Stochastic Dual Coordinate Ascent (SDCA) optimizer for

  linear models with L1 + L2 regularization. As global optimization objective is
  strongly-convex, the optimizer optimizes the dual objective at each step. The
  optimizer applies each update one example at a time. Examples are sampled
  uniformly, and the optimizer is learning rate free and enjoys linear convergence
  rate.

  Proximal Stochastic Dual Coordinate Ascent, Shalev-Shwartz, Shai; Zhang, Tong.
  2012 arXiv1211.2717S: http://arxiv.org/pdf/1211.2717v1.pdf

    Loss objective = \sum f_{i}(wx_{i}) + (l2 / 2) * |w|^2 + l1 * |w|

  Adding vs. Averaging in Distributed Primal-Dual Optimization.
  Chenxin Ma, Virginia Smith, Martin Jaggi, Michael I. Jordan, Peter Richtarik,
  Martin Takac http://arxiv.org/abs/1502.03508

  Args:
    sparse_example_indices: A list of `Tensor` objects of type `int64`.
      a list of vectors which contain example indices.
    sparse_feature_indices: A list with the same number of `Tensor` objects as `sparse_example_indices` of `Tensor` objects of type `int64`.
      a list of vectors which contain feature indices.
    sparse_feature_values: A list of `Tensor` objects of type `float32`.
      a list of vectors which contains feature value
      associated with each feature group.
    dense_features: A list of `Tensor` objects of type `float32`.
      a list of matrices which contains the dense feature values.
    example_weights: A `Tensor` of type `float32`.
      a vector which contains the weight associated with each
      example.
    example_labels: A `Tensor` of type `float32`.
      a vector which contains the label/target associated with each
      example.
    sparse_weights: A list with the same number of `Tensor` objects as `sparse_example_indices` of `Tensor` objects of type `float32`.
      a list of vectors where each value is the weight associated with
      a sparse feature group.
    dense_weights: A list with the same number of `Tensor` objects as `dense_features` of `Tensor` objects of type `float32`.
      a list of vectors where the values are the weights associated
      with a dense feature group.
    example_state_data: A `Tensor` of type `float32`.
      a list of vectors containing the example state data.
    loss_type: A `string` from: `"logistic_loss", "squared_loss", "hinge_loss", "smooth_hinge_loss"`.
      Type of the primal loss. Currently SdcaSolver supports logistic,
      squared and hinge losses.
    l1: A `float`. Symmetric l1 regularization strength.
    l2: A `float`. Symmetric l2 regularization strength.
    num_loss_partitions: An `int` that is `>= 1`.
      Number of partitions of the global loss function.
    num_inner_iterations: An `int` that is `>= 1`.
      Number of iterations per mini-batch.
    adaptative: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out_example_state_data, out_delta_sparse_weights, out_delta_dense_weights).
    out_example_state_data: A `Tensor` of type `float32`. a list of vectors containing the updated example state
      data.
    out_delta_sparse_weights: A list with the same number of `Tensor` objects as `sparse_example_indices` of `Tensor` objects of type `float32`. a list of vectors where each value is the delta
      weights associated with a sparse feature group.
    out_delta_dense_weights: A list with the same number of `Tensor` objects as `dense_features` of `Tensor` objects of type `float32`. a list of vectors where the values are the delta
      weights associated with a dense feature group.
  """
  result = _op_def_lib.apply_op("DistributedSdcaLargeBatchSolver",
                                sparse_example_indices=sparse_example_indices,
                                sparse_feature_indices=sparse_feature_indices,
                                sparse_feature_values=sparse_feature_values,
                                dense_features=dense_features,
                                example_weights=example_weights,
                                example_labels=example_labels,
                                sparse_weights=sparse_weights,
                                dense_weights=dense_weights,
                                example_state_data=example_state_data,
                                loss_type=loss_type, l1=l1, l2=l2,
                                num_loss_partitions=num_loss_partitions,
                                num_inner_iterations=num_inner_iterations,
                                adaptative=adaptative, name=name)
  return _DistributedSdcaLargeBatchSolverOutput._make(result)


ops.RegisterShape("DistributedSdcaLargeBatchSolver")(None)
_sdca_fprint_outputs = ["output"]


def sdca_fprint(input, name=None):
  r"""Computes fingerprints of the input strings.

  Args:
    input: A `Tensor` of type `string`.
      vector of strings to compute fingerprints on.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. vector containing the computed fingerprints.
  """
  result = _op_def_lib.apply_op("SdcaFprint", input=input, name=name)
  return result


ops.RegisterShape("SdcaFprint")(None)
_sdca_shrink_l1_outputs = [""]


def sdca_shrink_l1(weights, l1, l2, name=None):
  r"""Applies L1 regularization shrink step on the parameters.

  Args:
    weights: A list of `Tensor` objects of type mutable `float32`.
      a list of vectors where each value is the weight associated with a
      feature group.
    l1: A `float`. Symmetric l1 regularization strength.
    l2: A `float`.
      Symmetric l2 regularization strength. Should be a positive float.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("SdcaShrinkL1", weights=weights, l1=l1, l2=l2,
                                name=name)
  return result


ops.RegisterShape("SdcaShrinkL1")(None)
def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "DistributedSdcaLargeBatchSolver"
  input_arg {
    name: "sparse_example_indices"
    type: DT_INT64
    number_attr: "num_sparse_features"
  }
  input_arg {
    name: "sparse_feature_indices"
    type: DT_INT64
    number_attr: "num_sparse_features"
  }
  input_arg {
    name: "sparse_feature_values"
    type: DT_FLOAT
    number_attr: "num_sparse_features_with_values"
  }
  input_arg {
    name: "dense_features"
    type: DT_FLOAT
    number_attr: "num_dense_features"
  }
  input_arg {
    name: "example_weights"
    type: DT_FLOAT
  }
  input_arg {
    name: "example_labels"
    type: DT_FLOAT
  }
  input_arg {
    name: "sparse_weights"
    type: DT_FLOAT
    number_attr: "num_sparse_features"
  }
  input_arg {
    name: "dense_weights"
    type: DT_FLOAT
    number_attr: "num_dense_features"
  }
  input_arg {
    name: "example_state_data"
    type: DT_FLOAT
  }
  output_arg {
    name: "out_example_state_data"
    type: DT_FLOAT
  }
  output_arg {
    name: "out_delta_sparse_weights"
    type: DT_FLOAT
    number_attr: "num_sparse_features"
  }
  output_arg {
    name: "out_delta_dense_weights"
    type: DT_FLOAT
    number_attr: "num_dense_features"
  }
  attr {
    name: "loss_type"
    type: "string"
    allowed_values {
      list {
        s: "logistic_loss"
        s: "squared_loss"
        s: "hinge_loss"
        s: "smooth_hinge_loss"
      }
    }
  }
  attr {
    name: "adaptative"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "num_sparse_features"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "num_sparse_features_with_values"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "num_dense_features"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "l1"
    type: "float"
  }
  attr {
    name: "l2"
    type: "float"
  }
  attr {
    name: "num_loss_partitions"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "num_inner_iterations"
    type: "int"
    has_minimum: true
    minimum: 1
  }
}
op {
  name: "SdcaFprint"
  input_arg {
    name: "input"
    type: DT_STRING
  }
  output_arg {
    name: "output"
    type: DT_STRING
  }
}
op {
  name: "SdcaShrinkL1"
  input_arg {
    name: "weights"
    type: DT_FLOAT
    number_attr: "num_features"
    is_ref: true
  }
  attr {
    name: "num_features"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "l1"
    type: "float"
  }
  attr {
    name: "l2"
    type: "float"
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
