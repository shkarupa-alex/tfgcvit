# tfgcvit

Keras (TensorFlow v2) reimplementation of **Global Context Vision Transformer** models.

+ Based on [Official Pytorch implementation](https://github.com/nvlabs/gcvit).
+ Supports variable-shape inference for downstream tasks.
+ Contains pretrained weights converted from official ones.

## Installation

```bash
pip install tfgcvit
```

## Examples

Default usage (without preprocessing):

```python
from tfgcvit import GCViTTiny  # + 4 other variants and input preprocessing

model = GCViTTiny()  # by default will download imagenet-pretrained weights
model.compile(...)
model.fit(...)
```

Custom classification (with preprocessing):

```python
from keras import layers, models
from tfgcvit import GCViTTiny, preprocess_input

inputs = layers.Input(shape=(224, 224, 3), dtype='uint8')
outputs = layers.Lambda(preprocess_input)(inputs)
outputs = GCViTTiny(include_top=False)(outputs)
outputs = layers.Dense(100, activation='softmax')(outputs)

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(...)
model.fit(...)
```

## Differences

Code simplification:

- All input shapes automatically evaluated (not passed through a constructor like in PyTorch)
- Downsampling have been moved out from GCViTLayer layer to simplify feature extraction in downstream tasks.

Performance improvements:

- Layer normalization epsilon fixed at `1.001e-5` and inputs are casted to `float32` to use fused op implementation.
- Some layers have been refactored to use faster TF operations.
- A lot of reshapes/transposes have been removed. Most of the time internal representation is 4D-tensor.
- Relative index estimations moved to GCViTLayer layer level.

## Variable shapes

When using GCViT models with input shapes different from pretraining one, try to make height and width to be multiple
of `32 * window_size`. Otherwise, a lot of tensors will be padded, resulting in speed degradation.

## Evaluation

For correctness, `Tiny` and `Small` models (original and ported) tested
with [ImageNet-v2 test set](https://www.tensorflow.org/datasets/catalog/imagenet_v2).

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tfgcvit import GCViTTiny, preprocess_input


def _prepare(example, input_size=224, crop_pct=0.875):
    scale_size = tf.math.floor(input_size / crop_pct)

    image = example['image']

    shape = tf.shape(image)[:2]
    shape = tf.cast(shape, 'float32')
    shape *= scale_size / tf.reduce_min(shape)
    shape = tf.round(shape)
    shape = tf.cast(shape, 'int32')

    image = tf.image.resize(image, shape, method=tf.image.ResizeMethod.BICUBIC)
    image = tf.round(image)
    image = tf.clip_by_value(image, 0., 255.)
    image = tf.cast(image, 'uint8')

    pad_h, pad_w = tf.unstack((shape - input_size) // 2)
    image = image[pad_h:pad_h + input_size, pad_w:pad_w + input_size]

    image = preprocess_input(image)

    return image, example['label']


imagenet2 = tfds.load('imagenet_v2', split='test', shuffle_files=True)
imagenet2 = imagenet2.map(_prepare, num_parallel_calls=tf.data.AUTOTUNE)
imagenet2 = imagenet2.batch(8).prefetch(tf.data.AUTOTUNE)

model = GCViTTiny()
model.compile('sgd', 'sparse_categorical_crossentropy', ['accuracy', 'sparse_top_k_categorical_accuracy'])
history = model.evaluate(imagenet2)

print(history)
```

| name  | original acc@1 | ported acc@1 | original acc@5 | ported acc@5 |
|:-----:|:--------------:|:------------:|:--------------:|:------------:|
| Tiny  |     73.01      |    72.93     |     90.75      |    90.70     |
| Small |     73.39      |    73.46     |     91.09      |    91.14     |

The most metric differences comes from input data preprocessing (decoding, interpolation).
All layers outputs have been compared with original ones.
Maximum absolute difference among all layers is `8e-4`.
Most of them have maximum absolute difference less then `1e-5`.

## Citation

```
@article{hatamizadeh2022global,
  title={Global Context Vision Transformers},
  author={Hatamizadeh, Ali and Yin, Hongxu and Kautz, Jan and Molchanov, Pavlo},
  journal={arXiv preprint arXiv:2206.09959},
  year={2022}
}
```