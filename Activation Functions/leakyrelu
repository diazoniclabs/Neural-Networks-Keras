import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,33)
#x = np.random.randint(-10,10,100)
print(x)


def lrelu(z, alpha):
	return np.where(x > 0, x, x * alpha)


def dlrelu(x, alpha):
    return np.where(x > 0, 1, alpha)

y = lrelu(x,0.01)
d = dlrelu(x,0.01)
print(y)
print(d)

plt.figure(figsize=(20,5))
plt.scatter(x,y,s=50)
plt.scatter(x,d,c='r',s=50)
plt.show()
