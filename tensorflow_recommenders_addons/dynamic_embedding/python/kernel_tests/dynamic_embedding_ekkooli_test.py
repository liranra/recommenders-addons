# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""unit tests of variable
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import math
import numpy as np
import os
import six
import tempfile

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

def _type_converter(tf_type):
  mapper = {
    dtypes.int32: np.int32,
    dtypes.int64: np.int64,
    dtypes.float32: np.float,
    dtypes.float64: np.float64,
    dtypes.string: np.str,
    dtypes.half: np.float16,
    dtypes.int8: np.int8,
    dtypes.bool: np.bool,
  }
  return mapper[tf_type]

default_config = config_pb2.ConfigProto(
  allow_soft_placement=False,
  gpu_options=config_pb2.GPUOptions(allow_growth=True))

@test_util.run_all_in_graph_and_eager_modes
class VariableTest(test.TestCase):

  def test_variable(self):
    id = 0
    if test_util.is_gpu_available():
      dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 200]
      kv_list = [[dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int32],
                 [dtypes.int64, dtypes.half], [dtypes.int64, dtypes.int8]]
    else:
      dim_list = [1, 8, 16, 128]
      kv_list = [[dtypes.int32, dtypes.double], [dtypes.int32, dtypes.float32],
                 [dtypes.int32, dtypes.int32], [dtypes.int64, dtypes.double],
                 [dtypes.int64, dtypes.float32], [dtypes.int64, dtypes.int32],
                 [dtypes.int64, dtypes.int64], [dtypes.int64, dtypes.string],
                 [dtypes.int64, dtypes.int8], [dtypes.int64, dtypes.half],
                 [dtypes.string, dtypes.double],
                 [dtypes.string, dtypes.float32], [dtypes.string, dtypes.int32],
                 [dtypes.string, dtypes.int64], [dtypes.string, dtypes.int8],
                 [dtypes.string, dtypes.half]]

    def _convert(v, t):
      return np.array(v).astype(_type_converter(t))

    for (key_dtype, value_dtype), dim in itertools.product(kv_list, dim_list):
      id += 1
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        keys = constant_op.constant(
          np.array([0, 1, 2, 3]).astype(_type_converter(key_dtype)),
          key_dtype)
        values = constant_op.constant(
          _convert([[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype),
          value_dtype)
        table = de.get_variable('t1-' + str(id),
                                key_dtype=key_dtype,
                                value_dtype=value_dtype,
                                initializer=np.array([-1]).astype(
                                  _type_converter(value_dtype)),
                                dim=dim)
        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(table.upsert(keys, values))
        self.assertAllEqual(4, self.evaluate(table.size()))

        remove_keys = constant_op.constant(_convert([1, 5], key_dtype),
                                           key_dtype)
        self.evaluate(table.remove(remove_keys))
        self.assertAllEqual(3, self.evaluate(table.size()))

        remove_keys = constant_op.constant(_convert([0, 1, 5], key_dtype),
                                           key_dtype)
        output = table.lookup(remove_keys)
        self.assertAllEqual([3, dim], output.get_shape())

        result = self.evaluate(output)
        self.assertAllEqual(
          _convert([[0] * dim, [-1] * dim, [-1] * dim], value_dtype),
          _convert(result, value_dtype))

        exported_keys, exported_values = table.export()

        hot_exported_keys, hot_exported_values = table.export_hot_values()
        # exported data is in the order of the internal map, i.e. undefined
        sorted_keys = np.sort(self.evaluate(exported_keys))
        sorted_values = np.sort(self.evaluate(exported_values), axis=0)
        self.assertAllEqual(_convert([0, 2, 3], key_dtype),
                            _convert(sorted_keys, key_dtype))
        self.assertAllEqual(
          _convert([[0] * dim, [2] * dim, [3] * dim], value_dtype),
          _convert(sorted_values, value_dtype))

        del table

if __name__ == "__main__":
  test.main()
