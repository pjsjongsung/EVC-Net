# EVC-Net: Multi-scale V-Net with Conditional Random Fields for Brain Extraction
EVC-Net is a novel deep learning model with CRF for skulls stripping.
<img src='test_images/readme_figure.png' width=1000>

This repo provides a python code for training and testing the skull stripping models.

The main code is in src/models.py.

[Training example](examples/training_example.ipynb) and [Testing example](examples/testing_example.ipynb) are provided as an example of how to use the model.
***
## Required Packages
```
dipy
numpy
scikit-image
scipy
SimpleCRF
tensorflow
```

## Downloading
```
git clone git@github.com:pjsjongsung/EVC-Net.git
git lfs fetch
git lfs pull
```

## Simplest way to test
```
import src.models as models
import tensorflow as tf

input_dir = 'where/the/nifti_files/are'
output_dir = 'where/the/output_files/should_be'

model = tf.keras.models.load_model('trained_models/evnet/', custom_objects={'dice_coef': models.dice_coef})
models.test_model(model, input_dir, output_dir, batch_size=1, model_type='evcnet', crf_param=None)
```
