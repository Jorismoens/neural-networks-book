# Common loss class
import numpy as np


class Loss:

    # Calculates teh data and regularization lossen
    # given model output and ground truth values
    def calculate(self, output, y):

        #    Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        return data_loss
