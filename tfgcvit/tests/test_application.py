import numpy as np
import tensorflow as tf
import tfgcvit
from absl.testing import parameterized
from keras import layers, models
from keras.applications import imagenet_utils
from keras.utils import data_utils, image_utils

MODEL_LIST = [
    (tfgcvit.GCViTTiny, 224, 512),
    (tfgcvit.GCViTSmall, 224, 768),
    (tfgcvit.GCViTBase, 224, 1024)
]


class ApplicationTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(*MODEL_LIST)
    def test_application_base(self, app, *_):
        # Can be instantiated with default arguments
        model = app(weights=None)

        # Can be serialized and deserialized
        config = model.get_config()
        reconstructed_model = model.__class__.from_config(config)

        self.assertEqual(len(model.weights), len(reconstructed_model.weights))

    @parameterized.parameters(*MODEL_LIST)
    def test_application_notop(self, app, _, last_dim):
        output_shape = app(weights=None, include_top=False).output_shape
        self.assertLen(output_shape, 4)
        self.assertEqual(output_shape[-1], last_dim)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_pooling(self, app, _, last_dim):
        output_shape = app(weights=None, include_top=False, pooling='avg').output_shape
        self.assertLen(output_shape, 2)
        self.assertEqual(output_shape[-1], last_dim)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_input_1_channel(self, app, size, last_dim):
        input_shape = (size, size, 1)
        output_shape = app(weights=None, include_top=False, input_shape=input_shape).output_shape
        self.assertLen(output_shape, 4)
        self.assertEqual(output_shape[-1], last_dim)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_input_4_channels(self, app, size, last_dim):
        input_shape = (size, size, 4)
        output_shape = app(weights=None, include_top=False, input_shape=input_shape).output_shape
        self.assertLen(output_shape, 4)
        self.assertEqual(output_shape[-1], last_dim)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_weights_notop(self, app, size, last_dim):
        model = app(weights='imagenet', include_top=False)
        self.assertEqual(model.output_shape[-1], last_dim)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_predict(self, app, size, _):
        model = app(weights='imagenet')
        self.assertEqual(1000, model.output_shape[-1])

        test_image = data_utils.get_file(
            'elephant.jpg', 'https://storage.googleapis.com/tensorflow/keras-applications/tests/elephant.jpg')
        image = image_utils.load_img(test_image, target_size=(size, size), interpolation='bicubic')
        image = image_utils.img_to_array(image)[None, ...]

        image_ = tfgcvit.preprocess_input(image)
        preds = model.predict(image_)

        names = [p[1] for p in imagenet_utils.decode_predictions(preds, top=1)[0]]

        # Test correct label is in top 3 (weak correctness test).
        self.assertIn('African_elephant', names)

    # @parameterized.parameters(*MODEL_LIST)
    # def test_application_backbone(self, app, size, _):
    #     inputs = layers.Input(shape=(None, None, 3), dtype='uint8')
    #     outputs = layers.Lambda(tfgcvit.preprocess_input)(inputs)
    #     outputs = app(include_top=False)(outputs)
    #     outputs = layers.Conv2D(4, 3, padding='same', activation='softmax')(outputs)
    #     model = models.Model(inputs=inputs, outputs=outputs)
    #
    #     data = np.random.uniform(0., 255., size=(2, size * 2, size * 2, 3)).astype('uint8')
    #     result = model.predict(data)
    #
    #     self.assertTupleEqual(result.shape, (2, size * 2 // 32, size * 2 // 32, 4))


if __name__ == '__main__':
    tf.test.main()
