from math import log
import numpy as np

y_true = [1, 2]
y_pred = [[0.05, 0.95, 0], [0.1, 0.1,0.8]]

result = []
for i in y_pred:
  ce = -(log(np.max(i)))
  result.append(ce)

mean_ce = np.mean(result)
print(round(mean_ce,3))

# Maximum value range is from 0 to 2
# Output should be in Integer Encoding = y_true

# Predicted output is in Probabilities = 1.0

scce = tf.keras.losses.SparseCategoricalCrossentropy()
scce(y_true, y_pred).numpy()
