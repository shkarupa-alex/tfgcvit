import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from tfgcvit.pad import SymmetricPadding


@test_combinations.run_all_keras_modes
class TestSymmetricPadding(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            SymmetricPadding,
            kwargs={'padding': 1},
            input_shape=[2, 4, 5, 3],
            input_dtype='float32',
            expected_output_shape=[None, 6, 7, 3],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
