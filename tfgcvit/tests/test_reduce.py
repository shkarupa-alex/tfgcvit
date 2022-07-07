import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from tfgcvit.reduce import ReduceSize


@test_combinations.run_all_keras_modes
class TestReduceSize(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            ReduceSize,
            kwargs={'keep_channels': True},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 2, 3, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            ReduceSize,
            kwargs={'keep_channels': False},
            input_shape=[2, 7, 8, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 6],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
