"""
This library and script provide the function `predict(image) -> String`
that accepts an image of an object and returns the predicted class.
It uses the model at `assets/models/cropped.h5` and hard-coded category names.

When invoked directly as a script it predicts the class of the images
located in `assets/manual_validation`.
"""

import os

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("assets/models/cropped.h5")
#model = keras.models.load_model("assets/models/latest.h5")
categories = ['ajar', 'cat', 'feet', 'none', 'package']

def predict(image):
    image = image.astype('float32')#/255.0
    image = tf.image.resize(image, [240, 240])
    image = np.expand_dims(image, axis=0)

    result = model.predict(image)[0]
    score = tf.nn.softmax(result)
    pred_class = categories[np.argmax(score)]
    pred_prob = np.max(score)
    print(pred_class, pred_prob)

    return pred_class

def detect_cat(image):
    return predict(image) == 'cat'

if __name__ == '__main__':
    print(tf.__version__)

    folder = "assets/manual_validation"
    
    def test_image(path):
        image = Image.open(path)
        image = np.array(image)
        return predict(image)
    
    for f in os.listdir(folder):
        if not f.endswith('.jpg'):
            continue
    
        path = os.path.join(folder, f)
        print("#", path)
    
        res = test_image(path)
        print(" ", res)
