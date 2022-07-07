import numpy as np
import tensorflow as tf
from keras import layers
from keras.testing_infra import test_combinations
from keras.utils.generic_utils import register_keras_serializable
from tfgcvit.block import Block
from testing_utils import layer_multi_io_test


@register_keras_serializable('TFGCVit')
class BlockSqueeze(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [layers.InputSpec(ndim=4), layers.InputSpec(ndim=2, dtype='int32')]
        if self.global_query:
            self.input_spec.append(layers.InputSpec(ndim=4))

    def call(self, inputs, **kwargs):
        if self.global_query:
            inputs, relative_index, q = inputs

            return super().call([inputs, relative_index[0], q], **kwargs)
        else:
            inputs, relative_index = inputs

            return super().call([inputs, relative_index[0]], **kwargs)


@test_combinations.run_all_keras_modes
class TestBlock(test_combinations.TestCase):
    def test_layer(self):
        inputs = 10 * np.random.random((1, 7, 7, 96)) - 0.5
        index = np.zeros([1, 7 ** 4], 'int32')
        query = np.random.random((1, 3, 49, 32))

        layer_multi_io_test(
            BlockSqueeze,
            kwargs={'window_size': 7, 'num_heads': 3, 'global_query': False, 'mlp_ratio': 4., 'qkv_bias': True,
                    'qk_scale': None, 'drop': 0., 'attn_drop': 0., 'path_drop': 0.20000000298023224,
                    'layer_scale': None},
            input_datas=[inputs, index],
            input_dtypes=['float32', 'int32'],
            expected_output_shapes=[(None, 7, 7, 96)],
            expected_output_dtypes=['float32']
        )

        layer_multi_io_test(
            BlockSqueeze,
            kwargs={'window_size': 7, 'num_heads': 3, 'global_query': True, 'mlp_ratio': 4., 'qkv_bias': True,
                    'qk_scale': None, 'drop': 0., 'attn_drop': 0., 'path_drop': 0.20000000298023224,
                    'layer_scale': None},
            input_datas=[inputs, index, query],
            input_dtypes=['float32', 'int32', 'float32'],
            expected_output_shapes=[(None, 7, 7, 96)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
