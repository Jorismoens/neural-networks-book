import numpy as np


class Activation_ReLU:

    def __init__(self):
        self.output = None
        self.inputs = None
        self.dinputs = None

    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Derivative of ReLU is 0 before activation, slope is 1 after activation
        # Multiply with result up to now
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
