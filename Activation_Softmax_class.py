import numpy as np


class Activation_Softmax:

    def __init__(self):
        self.dinputs = None
        self.output = None

    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        #  Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Derivative of Softmax function
    # dS_ij/dz_ik = S_ij * delta_jk - S_ij * S_ik
    # With delta_jk the Kronecker delta
    def backward(self, dvalues):

        # TODO instead of line below from NN book, dinputs is declared in init, check if this compiles
        # self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
