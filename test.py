import tensorflow as tf
from keras.losses import binary_crossentropy
import numpy as np

print("Test\n ================================")

# Use valid probability values (0 < p < 1) and convert to tensors
y_true = tf.constant([0., 0., 1., 0.])  # true labels
y_pred = tf.constant([0.1, 0.2, 0.1, 0.1])  # predicted probabilities

loss = binary_crossentropy(y_true, y_pred)
print(f"Y_true: {y_true.numpy()}")
print(f"Y_pred: {y_pred.numpy()}")
print(f"Loss per sample: {loss.numpy()}")
print(f"Average loss: {tf.reduce_mean(loss).numpy()}")

print("End Test\n ================================")