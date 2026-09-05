import numpy as np
from sklearn.preprocessing import minmax_scale
import cv2
import math
import pandas as pd
from tqdm import tqdm

class Vision:
    def __init__(self, n_channels: int = 1, resolution: int = 256, kernel_size: int = 9) -> None:
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_kernels = int(pow(resolution, 2) / pow(kernel_size, 2))
        self.resolution = (resolution, resolution)
        self.__doc__ = ""

    def Load(self, path):
        if self.n_channels == 1:
            grayscale_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            grayscale_img = cv2.resize(grayscale_img, (self.resolution))
            grayscale_img = np.float64(grayscale_img)
            return grayscale_img
        elif self.n_channels == 3:
            rgb_img = cv2.imread(path, cv2.IMREAD_COLOR)
            rgb_img = cv2.resize(rgb_img, self.resolution)
            return rgb_img

    def ConvertToCurrent(self, img):
        img *= 255 / np.max(img)
        if self.n_channels == 1:
            img_as_current = minmax_scale(img, feature_range=(-75,-55))
        elif self.n_channels == 3:
            img_as_current = minmax_scale(img, feature_range=(-75,-55))
        return img_as_current
    
    def ToVector(self, img: np.ndarray):
        return img.flatten()

    def MakeTiles(self, img: np.ndarray) -> list:
        # Break down the img into tiles
        k_size = self.kernel_size
        tiles = []
        for x in range(0,img.shape[0],k_size):
            for y in range(0,img.shape[1],k_size):
                tiles.append(img[x:x+k_size,y:y+k_size])
        self.n_sensors = len(tiles)
        return tiles

    def split_quadrants(self, img: np.ndarray) -> list:
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        micro_tiles = [
            img[0:cY, 0:cX],
            img[0:cY, cX:w],
            img[cY:h, 0:cX],
            img[cY:h, cX:w]]
        return micro_tiles