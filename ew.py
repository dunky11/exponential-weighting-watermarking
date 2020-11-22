import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.ops import nn


class EWBase(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.is_ew_enabled = False

    def enable(self):
        self.is_ew_enabled = True

    def disable(self):
        self.is_ew_enabled = False

    def ew(self, theta):
        exp = tf.exp(tf.math.abs(theta) * self.t)
        numerator = exp
        denominator = tf.math.reduce_max(exp)
        return tf.math.multiply(numerator / denominator, theta)


class EWDense(EWBase):
    def __init__(self, units, t, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation
        self.t = t

    def build(self, input_shape):
        # ToDo change to glorot_normal since it's the default, but currently doesn't work with relu
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        if self.is_ew_enabled:
            out = tf.matmul(inputs, self.ew(self.w)) + self.b
        else:
            out = tf.matmul(inputs, self.w) + self.b
        if self.activation:
            return self.activation(out)
        return out


class EWConv2D(EWBase):
    def __init__(self, filters, kernel_size, t, strides=1, activation=None, padding="valid"):
        super().__init__()
        self.filters = filters
        self.t = t

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size,

        if isinstance(strides, int):
            self.strides = [strides, strides]
        elif isinstance(strides, tuple):
            self.strides = list(strides)
        else:
            self.strides = strides

        self.activation = activation

        if not padding.upper() in ["VALID", "SAME"]:
            raise Exception(
                f"padding must be either 'valid' or 'same', but '{padding}' was passed.")

        self.padding = padding.upper()
        self.t = t

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1],
                   input_shape[-1], self.filters),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.filters,), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        if self.is_ew_enabled:
            out = tf.nn.conv2d(inputs, self.ew(self.w),
                               strides=self.strides, padding=self.padding)
        else:
            out = tf.nn.conv2d(
                inputs, self.w, strides=self.strides, padding=self.padding)
        out = tf.nn.bias_add(out, self.b)
        if self.activation:
            return self.activation(out)
        return out
