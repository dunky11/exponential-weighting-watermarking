import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import random
from ew import EWBase, EWDense, EWConv2D

AUTOTUNE = tf.data.experimental.AUTOTUNE


def enable_ew(model):
    for layer in model.layers:
        if isinstance(layer, EWBase):
            layer.enable()

def disable_ew(model):
    for layer in model.layers:
        if isinstance(layer, EWBase):
            layer.disable()

def to_float(x, y):
    return tf.cast(x, tf.float32) / 255.0, y


dataset = tfds.load("mnist", as_supervised=True, split="train")
val_set = tfds.load("mnist", as_supervised=True, split="test")

dataset = dataset.map(to_float)
val_set = val_set.map(to_float)


# Generate the key set. In the paper they took a subset of the dataset and assigned random labels to them in order to combat query modification. However that altered the validation accuracy too much. For simplicity reasons we will just invert the pixels of a subset of the training dataset.


def invert(x, y):
    return (x * 2.0 - 1.0), y


key_set = dataset.take(128)
key_set = key_set.map(invert)
dataset = dataset.skip(128)


# An easy way to achieve a high accuracy on the key set is to overfit our model on the key set, since it doesn't have to generalize.


key_set = key_set.concatenate(key_set).concatenate(key_set).concatenate(key_set).concatenate(key_set).concatenate(key_set)

union = dataset.concatenate(key_set)

dataset = dataset.shuffle(2048).batch(128).prefetch(AUTOTUNE)
union = union.shuffle(2048).batch(128).prefetch(AUTOTUNE)
val_set = val_set.batch(128)


# t is the 'temperature' hyperparameter. The higher t is, the more the values of the weight matrix will get squeezed, 2.0 was used in the paper.


t = 2.0

model = keras.Sequential([
            EWConv2D(16, 3, t, padding="same", activation=keras.activations.relu),
            EWConv2D(32, 3, t, padding="same", strides=2, activation=keras.activations.relu),
            EWConv2D(64, 3, t, padding="same", strides=2, activation=keras.activations.relu),
            keras.layers.Flatten(),
            EWDense(10, activation=None, t=t)
        ])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["sparse_categorical_accuracy"])
model.build(input_shape=(None, 28, 28, 1))


# Train the model normally with exponential weighting disabled until it converges:

_ = model.fit(x=dataset, epochs=3, validation_data=val_set)


# Enable exponential weighting and train the model on the union of the dataset and the key set in order to embed the watermark:


enable_ew(model)
_ = model.fit(x=union, epochs=2, validation_data=val_set)


# Reset the optimizer. Disable exponential weighting and test the accuracy on the key set:


model.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
disable_ew(model)
_, key_acc = model.evaluate(key_set.batch(128))
_, val_acc = model.evaluate(val_set)

print(f"Watermark accuracy is {round(key_acc * 100, 2)}%.")
print(f"Validation set accuracy is {round(val_acc * 100, 2)}%.")


# If the watermark accuracy(key_acc) is above a predefined threshold, the model was watermarked by us.