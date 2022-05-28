import numpy as np


class FullyConnected:
    
	weights = np.array([])
	biases = np.array([])
	activation_function = None
    
	def __init__(self, nx, ny):
		if type(nx) is not int:
			raise TypeError("nx must be an integer")
		if nx < 1:
			raise ValueError("nx must be a positive integer")
		if type(ny) is not int:
			raise TypeError("ny must be an integer")
		if ny < 1:
			raise ValueError("ny must be a positive integer")
		self.weights = np.random.randn(ny, nx)
		self.biases = np.zeros((ny, 1))
  
	def forward(self, X):
		"""
		Calculates the forward propagation of the neuron
		"""
		self.input = X
		self.output = np.dot(self.weights, self.input) + self.biases
		return self.output

	def sigmoid(self, x):
		"""
		Calculates the sigmoid function
		"""
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self, x):
		"""
		Calculates the derivative of the sigmoid function
		"""
		return x * (1 - x)

	def softmax(self, x):
		"""
		Calculates the softmax function
		"""
		return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

	def softmax_derivative(self, x):
		"""
		Calculates the derivative of the softmax function
		"""
		return np.diagflat(x) - np.dot(x, x.T)

	def backward(self, X, Y, A):
		"""
		Calculates the derivative of the cost with respect to the weights and biases
  
		@param X: is a numpy.ndarray with shape (nx, m) that contains the input data
		@param Y: is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
		@param A: is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each exampleq
		"""
		self.m = X.shape[1]
		self.dZ = A - Y
		self.dW = np.dot(self.dZ, self.input.T) / self.m
		self.db = np.sum(self.dZ, axis=1, keepdims=True) / self.m
		self.dA = np.dot(self.weights.T, self.dZ)
		return self.dA, self.dW, self.db