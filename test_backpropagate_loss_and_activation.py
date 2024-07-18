from timeit import timeit
import numpy as np
import nnfs

from Neural_Network.Activation_Softmax_class import Activation_Softmax
from Neural_Network.Loss_Function_class import Loss_CategoricalCrossEntropy
from Neural_Network.Softmax_activation_and_CategoricalCrossentropy_loss import Combined_Activation_And_Loss

nnfs.init()

# We've implemented two ways of calculating the gradient of the Softmax activation and the Categorical Cross-Entropy.
# We do it separately in their own classes and have created a combined class to calculate their combined gradient
#  without the extra computation needed to calculate them separately. As the activation and the loss function are always
#  used after each other in our neural network we can make use of this to compute their combined gradient contribution
#  analytically. This turns out to be a much easier expression than both of them separately.

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

# Gradients of combined loss and activation class
softmax_loss = Combined_Activation_And_Loss()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs

# Gradients of separate loss and activation class
activation = Activation_Softmax()
activation.output = softmax_outputs
loss: Loss_CategoricalCrossEntropy = Loss_CategoricalCrossEntropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

print('Gradients: combined loss and activation: ')
print(dvalues1)
print('Gradients: separate loss and activation: ')
print(dvalues2)


# Results are the same

# Run again through timeit to measure performance gain
def f1():
    softmax_loss = Combined_Activation_And_Loss()
    softmax_loss.backward(softmax_outputs, class_targets)
    dvalues1 = softmax_loss.dinputs


def f2():
    activation = Activation_Softmax()
    activation.output = softmax_outputs
    loss: Loss_CategoricalCrossEntropy = Loss_CategoricalCrossEntropy()
    loss.backward(softmax_outputs, class_targets)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs


t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)
print(t2 / t1)

# Calculating the gradient combined is about 5 times faster
