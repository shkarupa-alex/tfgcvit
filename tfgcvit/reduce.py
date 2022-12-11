from keras import layers, models
from keras.saving.object_registration import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfgcvit.norm import LayerNorm
from tfgcvit.se import SE


@register_keras_serializable(package='TFGCVit')
class ReduceSize(layers.Layer):
    def __init__(self, keep_channels, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.keep_channels = keep_channels

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimensions of the inputs should be defined. Found `None`.')

        # noinspection PyAttributeOutsideInit
        self.norm2 = LayerNorm(name='norm2')

        # noinspection PyAttributeOutsideInit
        self.conv = [
            layers.DepthwiseConv2D(3, padding='same', activation='gelu', use_bias=False, name='conv/0'),
            SE(name='conv/2'),
            layers.Conv2D(channels, 1, use_bias=False, name='conv/3')
        ]

        # noinspection PyAttributeOutsideInit
        self.pad = layers.ZeroPadding2D(1)

        reduced_channels = channels * (2 - int(self.keep_channels))

        # noinspection PyAttributeOutsideInit
        self.reduce = layers.Conv2D(reduced_channels, 3, strides=2, use_bias=False, name='reduction')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        for layer in self.conv:
            residual = layer(residual)

        outputs = inputs + residual
        outputs = self.pad(outputs)
        outputs = self.reduce(outputs)
        outputs = self.norm2(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = self.pad.compute_output_shape(input_shape)
        output_shape = self.reduce.compute_output_shape(output_shape)

        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update({'keep_channels': self.keep_channels})

        return config
