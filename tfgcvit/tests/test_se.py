import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from tfgcvit.se import SE


@test_combinations.run_all_keras_modes
class TestSE(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            SE,
            kwargs={'expansion': 0.5},
            input_shape=[2, 4, 4, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            SE,
            kwargs={'expansion': 1.5},
            input_shape=[2, 4, 4, 3],
            input_dtype='float32',
            expected_output_shape=[None, 4, 4, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
