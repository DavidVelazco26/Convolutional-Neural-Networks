import numpy as np

"""A Single Neuron"""
inputs = [ 1 , 2 , 3 ]
weights = [ 0.2 , 0.8 , - 0.5 ]
bias = 2
output = (inputs[ 0 ] * weights[ 0 ] +
inputs[ 1 ] * weights[ 1 ] +
inputs[ 2 ] * weights[ 2 ] + bias)
print (output)

"""A Layer of Neurons"""
inputs = [ 1 , 2 , 3 , 2.5 ]
weights1 = [ 0.2 , 0.8 , - 0.5 , 1 ]
weights2 = [ 0.5 , - 0.91 , 0.26 , - 0.5 ]
weights3 = [ - 0.26 , - 0.27 , 0.17 , 0.87 ]
bias1 = 2
bias2 = 3
bias3 = 0.5
outputs = [
# Neuron 1:
inputs[ 0 ] * weights1[ 0 ] +
inputs[ 1 ] * weights1[ 1 ] +
inputs[ 2 ] * weights1[ 2 ] +
inputs[ 3 ] * weights1[ 3 ] + bias1,
# Neuron 2:
inputs[ 0 ] * weights2[ 0 ] +
inputs[ 1 ] * weights2[ 1 ] +
inputs[ 2 ] * weights2[ 2 ] +
inputs[ 3 ] * weights2[ 3 ] + bias2,
# Neuron 3:
inputs[ 0 ] * weights3[ 0 ] +
inputs[ 1 ] * weights3[ 1 ] +
inputs[ 2 ] * weights3[ 2 ] +
inputs[ 3 ] * weights3[ 3 ] + bias3]
print (outputs)

"""A Single Neuron with NumPy"""
inputs = [ 1.0 , 2.0 , 3.0 , 2.5 ]
weights = [ 0.2 , 0.8 , - 0.5 , 1.0 ]
bias = 2.0
outputs = np.dot(weights, inputs) + bias
print (outputs)
"""A Layer of Neurons with NumPy"""
inputs = [ 1.0 , 2.0 , 3.0 , 2.5 ]
weights = [[ 0.2 , 0.8 , - 0.5 , 1 ],
[ 0.5 , - 0.91 , 0.26 , - 0.5 ],
[ - 0.26 , - 0.27 , 0.17 , 0.87 ]]
biases = [ 2.0 , 3.0 , 0.5 ]
layer_outputs = np.dot(weights, inputs) + biases
print (layer_outputs)

