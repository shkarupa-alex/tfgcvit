import tensorflow as tf
from keras import initializers, layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfgcvit.drop import DropPath
from tfgcvit.mlp import MLP
from tfgcvit.norm import LayerNorm
from tfgcvit.winatt import WindowAttention
from tfgcvit.window import window_partition, window_reverse


@register_keras_serializable(package='TFGCVit')
class Block(layers.Layer):
    def __init__(self, window_size, num_heads, global_query, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., path_drop=0., layer_scale=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=1, dtype='int32')]
        if global_query:
            self.input_spec.append(layers.InputSpec(ndim=4, axes={1: num_heads}))

        self.window_size = window_size
        self.num_heads = num_heads
        self.global_query = global_query
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop
        self.layer_scale = layer_scale

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        channels = input_shape[0][-1]
        if channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        # noinspection PyAttributeOutsideInit
        self.norm1 = LayerNorm(name='norm1')

        # noinspection PyAttributeOutsideInit
        self.attn = WindowAttention(
            window_size=self.window_size, num_heads=self.num_heads, global_query=self.global_query,
            qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, attn_drop=self.attn_drop, proj_drop=self.drop, name='attn')

        # noinspection PyAttributeOutsideInit
        self.drop_path = DropPath(self.path_drop)

        # noinspection PyAttributeOutsideInit
        self.norm2 = LayerNorm(name='norm2')

        # noinspection PyAttributeOutsideInit
        self.mlp = MLP(ratio=self.mlp_ratio, dropout=self.drop, name='mlp')

        if self.layer_scale is not None:
            # noinspection PyAttributeOutsideInit
            self.gamma1 = self.add_weight(
                'gamma1',
                shape=[channels],
                initializer=initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype)

            # noinspection PyAttributeOutsideInit
            self.gamma2 = self.add_weight(
                'gamma2',
                shape=[channels],
                initializer=initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        if self.global_query:
            inputs, relative_index, query = inputs
        else:
            inputs, relative_index = inputs

        height, width = tf.unstack(tf.shape(inputs)[1:3])

        outputs = self.norm1(inputs)

        # Partition windows
        outputs = window_partition(outputs, height, width, self.window_size, self.compute_dtype)

        # W-MSA/SW-MSA
        if self.global_query:
            outputs = self.attn([outputs, relative_index, query])
        else:
            outputs = self.attn([outputs, relative_index])

        # Merge windows
        outputs = window_reverse(outputs, height, width, self.window_size, self.compute_dtype)

        # FFN
        if self.layer_scale is not None:
            outputs *= self.gamma1
        outputs = inputs + self.drop_path(outputs)
        residual = self.mlp(self.norm2(outputs))
        if self.layer_scale is not None:
            residual *= self.gamma2
        outputs += self.drop_path(residual)

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
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'drop': self.drop,
            'attn_drop': self.attn_drop,
            'path_drop': self.path_drop,
            'layer_scale': self.layer_scale
        })

        return config
