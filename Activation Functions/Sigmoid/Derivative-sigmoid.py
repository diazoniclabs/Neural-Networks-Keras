import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    s = 1/(1+np.exp(-x))   
    return s

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds

values = np.linspace(-10,10,100)

plt.plot(values, sigmoid(values), 'r',label = 'Sigmoid Function')
plt.plot(values, sigmoid_derivative(values), 'b',label = 'Derivative of Sigmoid')
plt.grid()
plt.title('Sigmoid and Sigmoid derivative functions')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.legend()
plt.show()
