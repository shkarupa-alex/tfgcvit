import numpy as np
import tensorflow as tf
from keras import layers
from keras.testing_infra import test_combinations
from keras.saving.object_registration import register_keras_serializable
from tfgcvit.winatt import WindowAttention
from testing_utils import layer_multi_io_test


@register_keras_serializable('TFGCVit')
class WindowAttentionSqueeze(WindowAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [layers.InputSpec(ndim=3), layers.InputSpec(ndim=2, dtype='int32')]
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
class TestWindowAttention(test_combinations.TestCase):
    def test_layer(self):
        inputs = 10 * np.random.random((1, 49, 96)) - 0.5
        index = np.zeros([1, 7 ** 4], 'int32')
        query = np.random.random((1, 3, 49, 32))

        layer_multi_io_test(
            WindowAttentionSqueeze,
            kwargs={
                'window_size': 7, 'num_heads': 3, 'global_query': False, 'qkv_bias': True, 'qk_scale': None,
                'attn_drop': 0., 'proj_drop': 0.},
            input_datas=[inputs, index],
            input_dtypes=['float32', 'int32'],
            expected_output_shapes=[(None, 49, 96)],
            expected_output_dtypes=['float32']
        )

        layer_multi_io_test(
            WindowAttentionSqueeze,
            kwargs={
                'window_size': 7, 'num_heads': 3, 'global_query': True, 'qkv_bias': True, 'qk_scale': None,
                'attn_drop': 0., 'proj_drop': 0.},
            input_datas=[inputs, index, query],
            input_dtypes=['float32', 'int32', 'float32'],
            expected_output_shapes=[(None, 49, 96)],
            expected_output_dtypes=['float32']
        )


if __name__ == '__main__':
    tf.test.main()
