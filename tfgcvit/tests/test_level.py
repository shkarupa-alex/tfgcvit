import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from tfgcvit.level import Level


@test_combinations.run_all_keras_modes
class TestLevel(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            Level,
            kwargs={'depth': 2, 'num_heads': 3, 'window_size': 7, 'keep_sizes': (False, False, False), 'mlp_ratio': 4.,
                    'qkv_bias': True, 'qk_scale': None, 'drop': 0., 'attn_drop': 0.,
                    'path_drop': [0.0, 0.0181818176060915], 'layer_scale': 1e-5},
            input_shape=[2, 56, 56, 96],
            input_dtype='float32',
            expected_output_shape=[None, 56, 56, 96],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
