"""
This script uses images in subfolders of `assets/train_cropped` to train a
classification model. The script outputs the resulting model at
`assets/models/latest.h5`. The training is interrupted early if the validation
loss stops improving. A performance graph is generated and displayed in a
window when possible, otherwise it is saved locally and the web server folder.
If the generated model is satisfactory, it should be moved to
`assets/model/cropped.h5` for use by other tools.
"""

import os
import sys
import pathlib
from random import randint

import numpy as np
import tensorflow as tf
import PIL
import matplotlib
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomZoom, RandomRotation, RandomFlip, RandomContrast
from tensorflow.keras.preprocessing import image_dataset_from_directory

print("tf", tf.__version__)
print("np", np.__version__)
print("plt", matplotlib.__version__, matplotlib.get_backend())

#data_dir = pathlib.Path("assets/train")
data_dir = pathlib.Path("assets/train_cropped")

seed = randint(0, 999_999_999)
print("seed:", seed)

batch_size = 32
img_height = 240
img_width = 240

train_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=seed,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)
print("classes:", class_names)

#plt.figure(figsize=(10, 10))
#for images, labels in train_ds.take(1):
#  for i in range(9):
#    ax = plt.subplot(3, 3, i + 1)
#    plt.imshow(images[i].numpy().astype("uint8"))
#    plt.title(class_names[labels[i]])
#    plt.axis("off")

#for image_batch, labels_batch in train_ds:
#  print(image_batch.shape)
#  print(labels_batch.shape)
#  break

train_ds = train_ds.cache().shuffle(256).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

augment_data = Sequential([
  RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
  RandomContrast(0.5), #, input_shape=(img_height, img_width, 3)),
  RandomZoom(0.2),
  RandomRotation(0.3),
])

model = Sequential([
  augment_data,
  Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=32):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
        else:
            print("Restoring best weights")
            self.model.set_weights(self.best_weights)


epochs=256
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[EarlyStoppingAtMinLoss()],
)

model.save("assets/models/latest.h5")
model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
#plt.grid()
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
#plt.grid()
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

if matplotlib.get_backend() != "agg":
    plt.show()
else:
    plt.savefig('assets/models/latest.png')
    plt.savefig('/var/www/html/pics/model.png')
