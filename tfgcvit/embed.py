from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='TFGCVit')
class PatchEmbedding(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.embed_dim = embed_dim

    @shape_type_conversion
    def build(self, input_shape):
        # noinspection PyAttributeOutsideInit
        self.pad = layers.ZeroPadding2D(1)

        # noinspection PyAttributeOutsideInit
        self.proj = layers.Conv2D(self.embed_dim, 3, strides=2, name='proj')

        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        outputs = self.pad(inputs)
        outputs = self.proj(outputs)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = self.pad.compute_output_shape(input_shape)
        output_shape = self.proj.compute_output_shape(output_shape)

        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim})

        return config
