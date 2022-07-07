import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from tfgcvit.norm import LayerNorm


@test_combinations.run_all_keras_modes
class TestLayerNorm(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            LayerNorm,
            kwargs={},
            input_shape=[2, 16, 16, 3],
            input_dtype='float32',
            expected_output_shape=[None, 16, 16, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            LayerNorm,
            kwargs={},
            input_shape=[2, 16, 3],
            input_dtype='float16',
            expected_output_shape=[None, 16, 3],
            expected_output_dtype='float16'
        )


if __name__ == '__main__':
    tf.test.main()
