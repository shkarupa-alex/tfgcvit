#!/usr/bin/env python3
import argparse
import gdown
import os
import tfgcvit
import torch

CHECKPOINTS = {
    'nano': 'https://drive.google.com/file/d/1Bfe63cGurkufL0mEUL05oposybbPnAom/view?usp=sharing',
    'micro': 'https://drive.google.com/file/d/15kt8VOXdAH_jF77g7pEPk-ZmZF13sHRd/view?usp=sharing',
    'tiny': 'https://drive.google.com/file/d/1C9lLgykooDF6CxZDFDnUqw5lEqoFgULh/view?usp=sharing',
    'small': 'https://drive.google.com/file/d/1bfEJQNutyDkPHAkgYcKWhjVTT_ZnYXp4/view?usp=sharing',
    'base': 'https://drive.google.com/file/d/1PFugO7dqfS-eubZi-yksM_FcYvUNjXBn/view?usp=sharing',
    'large': 'https://drive.google.com/file/d/1XDvFQrCkK-6QIpdLU1QrXWzjwnzNcH3E/view?usp=sharing'
}
TF_MODELS = {
    'nano': tfgcvit.GCViTNano,
    'micro': tfgcvit.GCViTMicro,
    'tiny': tfgcvit.GCViTTiny,
    'small': tfgcvit.GCViTSmall,
    'base': tfgcvit.GCViTBase,
    'large': tfgcvit.GCViTLarge
}


def convert_name(weight_name):
    weight_name = weight_name.replace(':0', '').replace('/', '.')
    weight_name = weight_name.replace('depthwise_kernel', 'weight').replace('kernel', 'weight')

    if 'gamma1' not in weight_name and 'gamma2' not in weight_name:
        weight_name = weight_name.replace('gamma', 'weight').replace('beta', 'bias')

    return weight_name


def convert_weight(weight_value, weight_name):
    if '/depthwise_kernel' in weight_name and 4 == len(weight_value.shape):
        return weight_value.transpose([2, 3, 0, 1])

    if '/kernel' in weight_name and 4 == len(weight_value.shape):
        return weight_value.transpose([2, 3, 1, 0])

    if '/kernel' in weight_name and 2 == len(weight_value.shape):
        return weight_value.T

    return weight_value


if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='GCViT Transformer weight conversion from PyTorch to TensorFlow')
    parser.add_argument(
        'model_type',
        type=str,
        choices=list(CHECKPOINTS.keys()),
        help='Model checkpoint to load')
    parser.add_argument(
        'out_path',
        type=str,
        help='Path to save TensorFlow model weights')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.out_path) and os.path.isdir(argv.out_path), 'Wrong output path'

    weights_path = os.path.join(argv.out_path, f'gcvit_{argv.model_type}.pth')
    gdown.download(url=CHECKPOINTS[argv.model_type], output=weights_path, quiet=False, fuzzy=True, resume=True)
    weights_torch = torch.load(weights_path, map_location=torch.device('cpu'))

    model = TF_MODELS[argv.model_type](weights=None)

    weights_tf = []
    for w in model.weights:
        name = convert_name(w.name)
        assert name in weights_torch['state_dict'], f'Can\'t find weight {name} ({w.name}) in checkpoint'

        weight = weights_torch['state_dict'].pop(name).numpy()
        weight = convert_weight(weight, w.name)
        assert w.shape == weight.shape, f'Weight {w.name} shape {w.shape} is not compatible with {weight.shape}'

        weights_tf.append(weight)

    model.set_weights(weights_tf)
    model.save_weights(weights_path.replace('.pth', '.h5'), save_format='h5')
