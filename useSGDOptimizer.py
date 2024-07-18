import nnfs
import numpy as np
from nnfs.datasets import spiral_data

from Layer_Dense_class import Layer_Dense
from activation_ReLU_class import Activation_ReLU
from Optimizer_SGD import Optimizer_SGD
from Softmax_activation_and_CategoricalCrossentropy_loss import Combined_Activation_And_Loss

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

# Create dense layer with 2 inputs and 64 outputs
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()

# Create second dense layer with 64 inputs and 3 outputs
dense2 = Layer_Dense(64, 3)

# Use the combined softmax activation and cross-entropy loss class
loss_activation = Combined_Activation_And_Loss()

learning_rate = 1.0
optimizer = Optimizer_SGD(learning_rate)

# Train in loop
for epoch in range(10001):

    # Walk through the network
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 500:
        print(f'epoch: {epoch}, '
              f'accuracy: {accuracy:.3f} '
              f'loss: {loss:.3f} '
              f'learning rate: {learning_rate:.3f}')

    # Perform backpropagation
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Using a variable learning rate that decreases when the model gets more accurate improves results
    learning_rate = 4*(1 - accuracy)**2

    # Update weights and biases
    optimizer.update_params(dense1, learning_rate)
    optimizer.update_params(dense2, learning_rate)


