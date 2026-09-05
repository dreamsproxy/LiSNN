import numpy as np
import cv2
from matplotlib import axis, pyplot as plt
from sklearn.preprocessing import minmax_scale
import math

def generate_grids(z, r, scale):
    #print(z, r, scale)
    x_lim = int(math.sqrt(r))
    y_lim = int(math.sqrt(r))
    x_c = minmax_scale(np.linspace(0.0, 1.0, x_lim), feature_range=(0.0, scale))
    y_c = minmax_scale(np.linspace(0.0, 1.0, y_lim), feature_range=(0.0, scale))
    coords = []
    for x in x_c:
        for y in y_c:
            
            coords.append(np.array([x, y, z]))
    coords = np.asarray(coords)

    return coords

def random_coords(neuron_count: int = 256, max_size: int = 0,
                x_lim: int = 256, y_lim: int = 256, z_lim: int = 256):
    if max_size >= 1:
        x_lim = max_size
        y_lim = max_size
        z_lim = max_size

    x_b = (0, x_lim)
    y_b = (0, y_lim)
    z_b = (0, z_lim)
    #Generate random coordinates within (size, size, size)
    x_coords = np.vstack(
        np.random.randint(x_b[0], high=x_b[1], size=(neuron_count)))
    y_coords = np.vstack(
        np.random.randint(y_b[0], high=y_b[1], size=(neuron_count)))
    z_coords = np.vstack(
        np.random.randint(z_b[0], high=z_b[1], size=(neuron_count)))

    neuron_coords = np.hstack((x_coords, y_coords, z_coords))

    return neuron_coords

def fibonacci(neuron_count: int = 256, size: int = 0):
    a = 0
    b = 1
    cache = []
    for i in range(1, neuron_count):
        c = a + b
        a = b
        b = c
        cache.append(b)
    cache = np.array(cache)
    return cache
    #relative_locs = minmax_scale(cache, feature_range=(0.0, 1.0))
    #if size > 0:
    #    relative_locs *= size
    #return relative_locs
