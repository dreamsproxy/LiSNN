import numpy as np

class non_linear:
    def GaborCorr(x: np.ndarray, y: np.ndarray, pearson_check: bool = False):
        """
        Gabor Szekely's
            Non-linear correlation
        """
        def double_center(x):
            centered = x.copy()
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    centered[i, j] = x[i, j] - np.mean(x[i, :]) - np.mean(x[:, j]) + np.mean(x)
            return centered

        def distance_covariance(x, y):
            N = len(x)
            dist_x = np.array(np.linalg.norm(x - x[:, None], axis=2))
            dist_y = np.array(np.linalg.norm(y - y[:, None], axis=2))
            centered_x = double_center(dist_x)
            centered_y = double_center(dist_y)
            calc = np.sum(centered_x * centered_y)
            return np.sqrt(calc / (N**2))

        def distance_variance(x):
            return distance_covariance(x, x)

        def distance_correlation(x, y):
            cov = distance_covariance(x, y)
            sd = np.sqrt(distance_variance(x) * distance_variance(y))
            return cov / sd

        # Compare with Pearson's r
        x = np.arange(-10, 11)
        y = x**2 + np.random.normal(0, 10, size=21)
        dist_corr = distance_correlation(x, y)
        if pearson_check:
            try:
                pearson_r = np.corrcoef(x, y)[0, 1]
                return np.ndarray([distance_correlation, np.corrcoef(x, y)[0,1]])
            except:
                print("Data size isn't equal!")
                return np.ndarray([distance_correlation, np.float32(0.0)])
        else:
            return dist_corr
