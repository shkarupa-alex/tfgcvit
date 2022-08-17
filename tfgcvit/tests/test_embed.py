import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from tfgcvit.embed import PatchEmbedding


@test_combinations.run_all_keras_modes
class TestPatchEmbedding(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            PatchEmbedding,
            kwargs={'embed_dim': 2},
            input_shape=[2, 12, 13, 3],
            input_dtype='float32',
            expected_output_shape=[None, 6, 7, 2],
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            PatchEmbedding,
            kwargs={'embed_dim': 2},
            input_shape=[2, 14, 15, 3],
            input_dtype='float32',
            expected_output_shape=[None, 7, 8, 2],
            expected_output_dtype='float32'
        )


if __name__ == '__main__':
    tf.test.main()
