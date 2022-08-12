from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from .pad import SymmetricPadding
from .se import SE


@register_keras_serializable(package='TFGCVit')
class FeatExtract(layers.Layer):
    def __init__(self, keep_size, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.keep_size = keep_size

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        # noinspection PyAttributeOutsideInit
        self.conv = [
            layers.DepthwiseConv2D(3, padding='same', activation='gelu', use_bias=False, name='conv/0'),
            SE(name='conv/2'),
            layers.Conv2D(channels, 1, use_bias=False, name='conv/3'),
        ]

        if not self.keep_size:
            # noinspection PyAttributeOutsideInit
            self.pad = SymmetricPadding(1)

            # noinspection PyAttributeOutsideInit
            self.pool = layers.MaxPool2D(3, strides=2)

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = inputs
        for layer in self.conv:
            outputs = layer(outputs)

        outputs += inputs

        if not self.keep_size:
            outputs = self.pad(outputs)
            outputs = self.pool(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.keep_size:
            return input_shape

        output_shape = self.pad.compute_output_shape(input_shape)
        output_shape = self.pool.compute_output_shape(output_shape)

        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update({'keep_size': self.keep_size})

        return config
