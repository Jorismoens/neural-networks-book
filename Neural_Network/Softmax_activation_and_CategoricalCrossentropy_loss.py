# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
from typing import Union, Any

import numpy as np

from Neural_Network.Activation_Softmax_class import Activation_Softmax
from Neural_Network.Loss_Function_class import Loss_CategoricalCrossEntropy


class Combined_Activation_And_Loss():

    # Creates activation and loss function objects
    def __init__(self):
        self.dinputs = None
        self.output = None
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs: Any, y_true: Any) -> float:
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        unnormalized_dinputs = dvalues.copy()
        # Calculate the gradient
        unnormalized_dinputs[range(samples), y_true] -= 1
        # Normalize the gradient
        self.dinputs = unnormalized_dinputs / samples
