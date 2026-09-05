import numpy as np

"""
IMPORTANT NOTE

All functions in the weight_init class returns a FILLED numpy ndarray object.
"""
def glorot(shape: tuple, dtype=np.float32) -> np.ndarray:
        """
        Glorot (Xavier) weight initializer.

        Parameters:
            - shape: Tuple specifying the shape of the weights.
            - dtype: Data type of the weights (default is np.float32).

        Returns:
            - weights: Initialized weights with the specified shape.
        """
        fan_in, fan_out = shape[0], shape[1]

        # Calculate the limit for the Glorot uniform initializer
        limit = np.sqrt(6 / (fan_in + fan_out))

        # Initialize weights from a uniform distribution within the limit
        weights = np.random.uniform(0, limit, size=shape).astype(dtype)
        
        return weights

def default(shape: tuple, dtype=np.float32):
    """
    Default: (0.50)
    Parameters:
        - Shape
        - dtype

    Returns:
        - weights, (ALL WEIGHTS AT 0.50)
    """
    weights = np.zeros(shape).astype(dtype)
    weights += np.float32(0.50)

    return weights