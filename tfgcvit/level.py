import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfgcvit.block import Block
from tfgcvit.extract import FeatExtract


@register_keras_serializable(package='TFGCVit')
class Level(layers.Layer):
    def __init__(self, depth, num_heads, window_size, keep_sizes, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., path_drop=0., layer_scale=None, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.keep_sizes = keep_sizes
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
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        path_drop = self.path_drop
        if not isinstance(self.path_drop, (list, tuple)):
            path_drop = [self.path_drop] * self.depth

        # noinspection PyAttributeOutsideInit
        self.blocks = [
            Block(window_size=self.window_size, num_heads=self.num_heads, global_query=bool(i % 2),
                  mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, drop=self.drop,
                  attn_drop=self.attn_drop, path_drop=path_drop[i], layer_scale=self.layer_scale, name=f'blocks/{i}')
            for i in range(self.depth)]

        # noinspection PyAttributeOutsideInit
        self.global_query = [
            FeatExtract(keep_size, name=f'q_global_gen/to_q_global/{i}')
            for i, keep_size in enumerate(self.keep_sizes)]

        # noinspection PyAttributeOutsideInit
        self.resize_query = layers.Resizing(self.window_size, self.window_size)

        super().build(input_shape)

    def relative_index(self):
        offset = tf.range(self.window_size)
        offset = tf.stack(tf.meshgrid(offset, offset, indexing='ij'), axis=0)
        offset = tf.reshape(offset, [2, -1])
        offset = offset[:, :, None] - offset[:, None]

        index = offset + (self.window_size - 1)
        index = index[0] * (2 * self.window_size - 1) + index[1]
        index = tf.reshape(index, [-1])

        return index

    def call(self, inputs, *args, **kwargs):
        height, width = tf.unstack(tf.shape(inputs)[1:3])
        h_pad = (self.window_size - height % self.window_size) % self.window_size
        w_pad = (self.window_size - width % self.window_size) % self.window_size
        outputs = tf.pad(inputs, [[0, 0], [0, h_pad], [0, w_pad], [0, 0]])
        padded_height, padded_width = height + h_pad, width + w_pad

        global_query = outputs
        for layer in self.global_query:
            global_query = layer(global_query)
        global_query = self.resize_query(global_query)

        input_length = padded_height * padded_width
        query_length = self.window_size ** 2

        global_query = tf.reshape(global_query, [-1, 1, query_length, self.num_heads, self.channels // self.num_heads])
        global_query = tf.transpose(global_query, [0, 1, 3, 2, 4])
        global_query = tf.tile(global_query, [1, input_length // query_length, 1, 1, 1])
        global_query = tf.reshape(global_query, [-1, self.num_heads, query_length, self.channels // self.num_heads])

        relative_index = self.relative_index()
        for i, b in enumerate(self.blocks):
            if i % 2:
                outputs = b([outputs, relative_index, global_query])
            else:
                outputs = b([outputs, relative_index])

        outputs = outputs[:, :height, :width, ...]
        outputs.set_shape(inputs.shape)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'depth': self.depth,
            'num_heads': self.num_heads,
            'window_size': self.window_size,
            'keep_sizes': self.keep_sizes,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'drop': self.drop,
            'attn_drop': self.attn_drop,
            'path_drop': self.path_drop,
            'layer_scale': self.layer_scale
        })

        return config
