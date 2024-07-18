# Stochastic Gradient Descent optimizer
from Neural_Network.Layer_Dense_class import Layer_Dense


class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1 is the default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer: Layer_Dense, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        layer.weights += - self.learning_rate * layer.dweights
        layer.biases += - self.learning_rate * layer.dbiases
