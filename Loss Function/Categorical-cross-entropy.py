from math import log
import numpy as np

y_true = [0, 1, 0]
y_pred = [0.05, 0.95, 0]


def categorical_crossentropy(target, output):
    output = np.clip(output, 0.0005, 1 - 0.0005)
    # Give small value else, will get log of zero error
    return np.sum(target * -np.log(output), axis=-1)

print(round(categorical_crossentropy(y_true,y_pred),3))

import tensorflow as tf
cce = tf.keras.losses.CategoricalCrossentropy()
print(round(cce(y_true, y_pred).numpy(),3))
