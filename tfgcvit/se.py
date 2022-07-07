from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFGCVit')
class SE(layers.Layer):
    def __init__(self, expansion=0.25, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.expansion = expansion

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        squeeze_filters = max(1, int(channels * self.expansion))

        # noinspection PyAttributeOutsideInit
        self.fc = [
            layers.GlobalAvgPool2D(keepdims=True),
            layers.Dense(squeeze_filters, activation='gelu', use_bias=False, name='fc/0'),
            layers.Dense(channels, activation='sigmoid', use_bias=False, name='fc/2')
        ]

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = inputs
        for layer in self.fc:
            outputs = layer(outputs)

        outputs *= inputs

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'expansion': self.expansion})

        return config
