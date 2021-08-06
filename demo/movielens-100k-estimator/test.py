import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.training import optimizer

op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)

if isinstance(op, optimizer.Optimizer):
    print(2)
elif isinstance(op, optimizer_v2.OptimizerV2):
    print(1)
