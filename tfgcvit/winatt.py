import numpy as np
import tensorflow as tf
from keras import initializers, layers
from keras.utils.control_flow_util import smart_cond
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFGCVit')
class WindowAttention(layers.Layer):
    def __init__(self, window_size, num_heads, global_query, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=3), layers.InputSpec(ndim=1, dtype='int32')]
        if global_query:
            self.input_spec.append(layers.InputSpec(ndim=4, axes={1: num_heads}))

        self.window_size = window_size
        self.num_heads = num_heads
        self.global_query = global_query
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.channels = input_shape[0][-1]
        if self.channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        self.qkv_size = 3 - int(self.global_query)

        # noinspection PyAttributeOutsideInit
        self.qkv = layers.Dense(self.channels * self.qkv_size, use_bias=self.qkv_bias, name='qkv')

        # noinspection PyAttributeOutsideInit
        self.scale = self.qk_scale or (self.channels // self.num_heads) ** -0.5

        # noinspection PyAttributeOutsideInit
        self.relative_bias = self.add_weight(
            'relative_position_bias_table',
            shape=[(2 * self.window_size - 1) ** 2, self.num_heads],
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype)

        # noinspection PyAttributeOutsideInit
        self.drop_attn = layers.Dropout(self.attn_drop)

        # noinspection PyAttributeOutsideInit
        self.proj = layers.Dense(self.channels, name='proj')

        # noinspection PyAttributeOutsideInit
        self.drop_proj = layers.Dropout(self.proj_drop)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if self.global_query:
            inputs, relative_index, q = inputs
        else:
            inputs, relative_index = inputs
        length = tf.shape(inputs)[1]

        qkv = self.qkv(inputs)
        qkv = tf.reshape(qkv, [-1, length, self.qkv_size, self.num_heads, self.channels // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])

        if self.global_query:
            k, v = tf.unstack(qkv, 2)
        else:
            q, k, v = tf.unstack(qkv, 3)
        q *= self.scale
        attn = tf.matmul(q, k, transpose_b=True)

        bias = tf.gather(self.relative_bias, relative_index)
        bias = tf.reshape(bias, [self.window_size ** 2, self.window_size ** 2, -1])
        bias = tf.transpose(bias, perm=[2, 0, 1])
        attn = attn + bias[None]

        attn = tf.nn.softmax(attn)
        attn = self.drop_attn(attn)

        outputs = tf.transpose(tf.matmul(attn, v), perm=[0, 2, 1, 3])
        outputs = tf.reshape(outputs, [-1, length, self.channels])

        outputs = self.proj(outputs)
        outputs = self.drop_proj(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'window_size': self.window_size,
            'num_heads': self.num_heads,
            'global_query': self.global_query,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'attn_drop': self.attn_drop,
            'proj_drop': self.proj_drop
        })

        return config
