import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10,10,100)
z = 1/(1 + np.exp(-x)) 
plt.plot(x,z,label = 'sigmoid')
plt.legend()
plt.show()


import tensorflow as tf
x = np.linspace(-10,10,100)
y_sigmoid = tf.nn.sigmoid(x)
plt.plot(x,y_sigmoid,label = 'sigmoid')
plt.legend()
plt.show()
