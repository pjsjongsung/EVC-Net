# EVC-Net: Multi-scale V-Net with Conditional Random Fields for Brain Extraction

This repo provides a python code for training and testing the skull stripping models.

The main code is in src/models.py.

Training and testing examples are given as a jupyter notebook in examples.

# Simplest way to test
```
import src.models as models
import tensorflow as tf

input_dir = 'where/the/nifti_files/are'
output_dir = 'where/the/output_files/should_be'

model = tf.keras.models.load_model('trained_models/evnet/', custom_objects={'dice_coef': models.dice_coef})
models.test_model(model, input_dir, output_dir, batch_size=1, model_type='evcnet', crf_param=None)
```