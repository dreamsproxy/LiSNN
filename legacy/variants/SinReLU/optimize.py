import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from numba import njit
from w_init import glorot
from scipy.spatial.distance import euclidean

class gradient_descent:
    def dtw_distance(s1: np.ndarray, s2:np.ndarray):
        """
        Calculate the Dynamic Time Warping distance between two sequences s1 and s2.
        
        Parameters:
            s1, s2: numpy arrays
            
        Returns:
            dtw_dist: float, the DTW distance between s1 and s2
        """
        s2 *= np.e
        # Length of the two sequences
        n, m = len(s1), len(s2)
        s1 = (s1 - s1.min()) / (s1.max() - s1.min())
        s2 = (s2 - s2.min()) / (s2.max() - s2.min())

        # Initialize the DTW matrix
        dtw_matrix = np.zeros((n + 1, m + 1))
        for i in range(n + 1):
            for j in range(m + 1):
                dtw_matrix[i, j] = np.inf
        
        # Set the origin to 0
        dtw_matrix[0, 0] = 0

        # Fill the DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = euclidean(np.absolute(s1), np.absolute(s2))  # Euclidean distance or any other cost function
                dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # Insertion
                                            dtw_matrix[i, j - 1],    # Deletion
                                            dtw_matrix[i - 1, j - 1])  # Match
                
        # Return the DTW distance
        return dtw_matrix[n, m]

    def cosine_similarity_loss(y_true, y_pred):
        """
        Calculate the cosine similarity loss between y_true and y_pred.
        
        Parameters:
            y_true, y_pred: numpy arrays, sequences of elements
            
        Returns:
            cosine_loss: float, cosine similarity loss
        """
        cosine_loss = 1 - cosine(y_true, y_pred)
        return cosine_loss

    def rmse_cost(y_true, y_pred):
        """
        Calculate the RMSE cost using DTW Distance and Cosine Similarity Loss.
        
        Parameters:
            y_true, y_pred: numpy arrays, sequences of elements
            
        Returns:
            rmse: float, RMSE cost
        """
        # Calculate DTW distance
        dtw_dist = gradient_descent.dtw_distance(y_true, y_pred)
        # Calculate Cosine Similarity Loss
        cosine_loss = gradient_descent.cosine_similarity_loss(y_true, y_pred)
        
        # Combine DTW Distance and Cosine Similarity Loss using RMSE
        rmse = np.sqrt(np.mean(np.square([dtw_dist, cosine_loss])))
        """try:
            # Calculate DTW distance
            dtw_dist = gradient_descent.dtw_distance(y_true, y_pred)
            # Calculate Cosine Similarity Loss
            cosine_loss = gradient_descent.cosine_similarity_loss(y_true, y_pred)
            
            # Combine DTW Distance and Cosine Similarity Loss using RMSE
            rmse = np.sqrt(np.mean(np.square([dtw_dist, cosine_loss])))
        except:
            # Calculate Cosine Similarity Loss
            cosine_loss = gradient_descent.cosine_similarity_loss(y_true, y_pred)
            # Combine DTW Distance and Cosine Similarity Loss using RMSE
            rmse = np.sqrt(np.mean(np.square([cosine_loss])))
            
        """
        return rmse

    def compute_gradients(output_recieved: np.ndarray, weights: np.ndarray, y_true):
        # Forward pass
        y_pred = output_recieved
        
        # Compute loss
        error = gradient_descent.rmse_cost(y_true, y_pred)
        
        # Initialize gradients
        gradients = np.zeros_like(weights)
        # Loop over each weight
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                # Perturb the weight
                weights[i, j] += 1e-6
                
                # Compute forward pass with the perturbed weight
                y_pred_perturbed = output_recieved
                
                # Compute loss with the perturbed weight
                error_perturbed = gradient_descent.rmse_cost(y_true, y_pred_perturbed)
                
                # Compute gradient using central difference
                gradients[i, j] = (error_perturbed - error) / 1e-6
                
                # Reset the weight to its original value
                weights[i, j] -= 1e-6
        
        return gradients, error

    # Gradient Descent Optimization
    def optimize_weights(inputs, weights, y_true, learning_rate=0.01, num_iterations=1):
        for _ in range(num_iterations):
            gradients, error = gradient_descent.compute_gradients(inputs, weights, y_true)
            weights -= learning_rate * gradients
            weights = (weights-np.min(weights))/(np.max(weights)-np.min(weights))
        return weights, error

    def example():
        # Example usage:
        y_true = np.array([1, 2, 3, 4, 5])
        y_true = (y_true-np.min(y_true))/(np.max(y_true)-np.min(y_true))
        y_pred = np.array([1, 3, 5, 7, 9])
        y_pred = (y_pred-np.min(y_pred))/(np.max(y_pred)-np.min(y_pred))

        #weights = (weights-np.min(weights))/(np.max(weights)-np.min(weights))
        #gradients = compute_gradients(y_pred, weights, y_true)

        cost = rmse_cost(y_true, y_pred)
        weights = glorot((5, 5))
        new_weights = optimize_weights(y_pred, weights, y_true)
