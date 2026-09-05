import numpy as np
import matplotlib.pyplot as plt
from numba import njit
class SineReLU:
    def __init__(self, threshold=0.0, dt=0.1, decay=0.99):
        self.threshold = threshold
        self.dt = dt
        self.decay = decay
        self.state = 0.0  # Initial state

    def discrete(self, input_data):
        # Increment state by input data multiplied by time step (dv/dt)
        self.state += input_data * self.dt
        # Apply decay to the state
        self.state *= self.decay
        self.dt += 1

    def activate(self, input_data):
        self.discrete(input_data)
        # Apply Sine ReLU activation function
        output = np.sin(self.state)
        if self.state > self.threshold:
            fire = True
        else:
            output = np.float32(0.0)
            fire = False

        return output, fire

from numba import njit
class SeLU:
    @njit
    def activate(x, alpha = 1.67326, lambda_ = 1.0507, decay_rate=0.95) -> np.float32:
        """
        Scaled Exponential Linear Unit

        Parameters:
            - input array
            - alpha
            - lambda_ scaler
            - decay_rate

        Returns:
            - sum of elements (element-wise)
        
        # Example usage
        x = np.array([-1, 0, 1, -2, 3])
        print("Input:", x)
        print("SELU output with decay:", selu_with_decay(x))
        """
        selu_output = lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))
        # Apply decay to negative values
        selu_output = np.where(x < 0, decay_rate * selu_output, selu_output)
        selu_output = np.float32(np.sum(selu_output))
        return selu_output
