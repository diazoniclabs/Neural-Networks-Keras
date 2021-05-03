# ReLu : Rectified Linear Unit
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,33)
#x = np.random.randint(-10,10,100)
print(x)

def relu(x):
  return x * (x > 0)

def d_relu(x):
  return 1 * (x > 0)


y = relu(x)
d = d_relu(x)
print(y)
print(d)

plt.plot(x,y)
plt.scatter(x,d,c='r')
plt.show()
