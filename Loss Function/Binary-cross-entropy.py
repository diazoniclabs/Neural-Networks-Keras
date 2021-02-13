from math import log
import numpy as np

def cross_entropy(a,p):
  return -((a*log(p))+((1-a)*log(1-p)))

a = [1,1,1,0,0,0]  # Actual Output
p = [0.9,0.85,0.75,0.2,0.2,0.1]   # Predicted Output

result = []
for i in range(len(a)):
  ce = cross_entropy(a[i],p[i])
  result.append(ce)

mean_ce = np.mean(result)
print(round(mean_ce,3))


import tensorflow as tf
bce = tf.keras.losses.BinaryCrossentropy()
op = round(bce(a, p).numpy(),3)
print(op)
