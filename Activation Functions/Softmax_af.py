#NUMPY IMPLEMENTATION OF SOFTMAX

import numpy as np

def softmax(inputs):
    return np.exp(inputs) / float(sum(np.exp(inputs)))
 
softmax_inputs = [1.3,5.1,2.2,0.7,1.1]
op = np.round(softmax(softmax_inputs),2)
print (op)

# tf.nn.softmax(softmax_inputs)
