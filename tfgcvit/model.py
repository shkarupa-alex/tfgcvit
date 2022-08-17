import numpy as np
import tensorflow as tf
from keras import backend, layers, models
from keras.applications import imagenet_utils
from keras.utils import data_utils, layer_utils
from tfgcvit.embed import PatchEmbedding
from tfgcvit.norm import LayerNorm
from tfgcvit.reduce import ReduceSize
from tfgcvit.level import Level

BASE_URL = 'https://github.com/shkarupa-alex/tfgcvit/releases/download/{}/gcvit_{}.h5'
WEIGHT_URLS = {
    'gcvit_nano': BASE_URL.format('1.0.0', 'nano'),
    'gcvit_micro': BASE_URL.format('1.0.0', 'micro'),
    'gcvit_tiny': BASE_URL.format('1.0.0', 'tiny'),
    'gcvit_small': BASE_URL.format('1.0.0', 'small'),
    'gcvit_base': BASE_URL.format('1.0.0', 'base'),
}
WEIGHT_HASHES = {
    'gcvit_nano': '752926536d36707415c8b17d819fb1bfc48d22fd878edde1f622c76bfe23f690',
    'gcvit_micro': 'fcea210cd00d79de3fc681ddaad965ca3601077a27db256d4aacddc1154b5517',
    'gcvit_tiny': 'b55e8de5e64174619bf1ffeb11ea2d9b553ce527d6aa4370f5ade875c6e7b1f5',
    'gcvit_small': '0d9755ce464c8f4eece85493697c694ea616036d41136a602204c2fddec67b1b',
    'gcvit_base': 'bcf1dd6a59f2ef12b0aa657f30aef0ed67bb6d17e9e91186c03e4da651b28b10'
}


def GCViT(
        window_size, embed_dim, depths, num_heads,
        drop_rate=0., mlp_ratio=3., qkv_bias=True, qk_scale=None, attn_drop=0., path_drop=0.1, layer_scale=None,
        model_name='gcvit', include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None,
        classes=1000, classifier_activation='softmax'):
    """Instantiates the GCViT Transformer architecture.

    Args:
      window_size: window partition size.
      embed_dim: patch embedding dimension.
      depths: depth of each level.
      num_heads: number of attention heads.
      drop_rate: dropout rate.
      mlp_ratio: ratio of mlp hidden units to embedding units.
      qkv_bias: whether to add a learnable bias to query, key, value.
      qk_scale: override default qk scale of head_dim ** -0.5 if set
      attn_drop: attention dropout rate
      path_drop: stochastic depth rate
      layer_scale: initial value for learnable block output scaling
      model_name: model name.
      include_top: whether to include the fully-connected layer at the top of the network.
      weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet or ImageNet 21k), or the
        path to the weights file to be loaded.
      input_tensor: tensor (i.e. output of `layers.Input()`) to use as image input for the model.
      input_shape: shape tuple without batch dimension. Used to create input layer if `input_tensor` not provided.
      pooling: optional pooling mode for feature extraction when `include_top` is `False`.
        - `None` means that the output of the model will be the 3D tensor output of the last layer.
        - `avg` means that global average pooling will be applied to the output of the last layer, and thus the output
          of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be specified if `include_top` is True.
      classifier_activation: the activation function to use on the "top" layer. Ignored unless `include_top=True`.
        When loading pretrained weights, `classifier_activation` can only be `None` or `"softmax"`.

    Returns:
      A `keras.Model` instance.
    """
    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError('The `weights` argument should be either `None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes not in {1000, 21841}:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` as true, '
                         '`classes` should be 1000 or 21841 depending on model type')

    if input_tensor is not None:
        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(f'Expecting `input_tensor` to be a symbolic tensor instance. '
                             f'Got {input_tensor} of type {type(input_tensor)}')

    if input_tensor is not None:
        tensor_shape = backend.int_shape(input_tensor)[1:]
        if input_shape and tensor_shape != input_shape:
            raise ValueError('Shape of `input_tensor` should equals to `input_shape` if both provided.')
        else:
            input_shape = tensor_shape

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format='channel_last',
        require_flatten=False,
        weights=weights)

    if input_tensor is not None:
        if backend.is_keras_tensor(input_tensor):
            image = input_tensor
        else:
            image = layers.Input(tensor=input_tensor, shape=input_shape, dtype='float32')
    else:
        image = layers.Input(shape=input_shape)

    # Define model pipeline
    x = PatchEmbedding(embed_dim=embed_dim, name='patch_embed')(image)
    x = LayerNorm(name='patch_embed/conv_down/norm1')(x)
    x = ReduceSize(True, name='patch_embed/conv_down')(x)
    x = layers.Dropout(drop_rate, name='pos_drop')(x)

    path_drops = np.linspace(0., path_drop, sum(depths))
    keep_sizes = [
        (False, False, False),
        (False, False),
        (True,),
        (True,),
    ]

    for i in range(len(depths)):
        path_drop = path_drops[sum(depths[:i]):sum(depths[:i + 1])].tolist()
        not_last = i != len(depths) - 1

        x = Level(depth=depths[i], num_heads=num_heads[i], window_size=window_size[i], keep_sizes=keep_sizes[i],
                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop,
                  path_drop=path_drop, layer_scale=layer_scale, name=f'levels/{i}')(x)
        if not_last:
            x = LayerNorm(name=f'levels/{i}/downsample/norm1')(x)
            x = ReduceSize(False, name=f'levels/{i}/downsample')(x)

    x = LayerNorm(name='norm')(x)

    if include_top or pooling in {None, 'avg'}:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    else:
        raise ValueError(f'Expecting pooling to be one of None/avg/max. Found: {pooling}')

    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, name='head')(x)
    x = layers.Activation(classifier_activation, dtype='float32', name='pred')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = image

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if 'imagenet' == weights and model_name in WEIGHT_URLS:
        weights_url = WEIGHT_URLS[model_name]
        weights_hash = WEIGHT_HASHES[model_name]
        weights_path = data_utils.get_file(origin=weights_url, file_hash=weights_hash, cache_subdir='tfgcvit')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if include_top:
        return model

    last_layer = 'norm'
    if pooling == 'avg':
        last_layer = 'avg_pool'
    elif pooling == 'max':
        last_layer = 'max_pool'

    outputs = model.get_layer(name=last_layer).output
    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    return model


def GCViTNano(model_name='gcvit_nano', window_size=(7, 7, 14, 7), embed_dim=64, depths=(2, 2, 6, 2),
              num_heads=(2, 4, 8, 16), path_drop=0.2, weights='imagenet', **kwargs):
    return GCViT(model_name=model_name, window_size=window_size, embed_dim=embed_dim, depths=depths,
                 num_heads=num_heads, path_drop=path_drop, weights=weights, **kwargs)


def GCViTMicro(model_name='gcvit_micro', window_size=(7, 7, 14, 7), embed_dim=64, depths=(3, 4, 6, 5),
               num_heads=(2, 4, 8, 16), path_drop=0.2, weights='imagenet', **kwargs):
    return GCViT(model_name=model_name, window_size=window_size, embed_dim=embed_dim, depths=depths,
                 num_heads=num_heads, path_drop=path_drop, weights=weights, **kwargs)


def GCViTTiny(model_name='gcvit_tiny', window_size=(7, 7, 14, 7), embed_dim=64, depths=(3, 4, 19, 5),
              num_heads=(2, 4, 8, 16), path_drop=0.2, weights='imagenet', **kwargs):
    return GCViT(model_name=model_name, window_size=window_size, embed_dim=embed_dim, depths=depths,
                 num_heads=num_heads, path_drop=path_drop, weights=weights, **kwargs)


def GCViTSmall(model_name='gcvit_small', window_size=(7, 7, 14, 7), embed_dim=96, depths=(3, 4, 19, 5),
               num_heads=(3, 6, 12, 24), mlp_ratio=2., path_drop=0.3, layer_scale=1e-5, weights='imagenet', **kwargs):
    return GCViT(model_name=model_name, window_size=window_size, embed_dim=embed_dim, depths=depths,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, path_drop=path_drop, layer_scale=layer_scale,
                 weights=weights, **kwargs)


def GCViTBase(model_name='gcvit_base', window_size=(7, 7, 14, 7), embed_dim=128, depths=(3, 4, 19, 5),
              num_heads=(4, 8, 16, 32), mlp_ratio=2., path_drop=0.5, layer_scale=1e-5, weights='imagenet', **kwargs):
    return GCViT(model_name=model_name, window_size=window_size, embed_dim=embed_dim, depths=depths,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, path_drop=path_drop, layer_scale=layer_scale,
                 weights=weights, **kwargs)
