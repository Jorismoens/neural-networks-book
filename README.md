Work-through of the projects in 'Neural Networks from scratch with Python" by Harrison Kinsley & Daniel Kukieła

In this book a neural network is build up from the ground in raw Python. None of the normal tools like PyTorch are used. Instead we build the entire toolkit ourselves.
This is done to get a good understanding for the underlying principles.

In simple_optimization.py a small neural network is set up and we try to categorize a simple dataset with it. 
The network is trained by making small, random adjustments to the weights and biases, and keeping the best set of weights and biases to iterate this process.
This works reasonably well for the simple dataset and a accuracy of 93% is achieved. However, trying this network on a more complicated dataset doesn't yield any usefull results.

Next, we turn to back-propagation to update the weights and biases more intelligently. To achieve this we calculate the derivative of the final loss with respect to each weight and bias.
These derivatives are calculated at every operation while moving forward with the network, and saved for later use.
