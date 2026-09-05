import numpy as np
from sklearn.preprocessing import minmax_scale
import cv2
import math
import pandas as pd
from tqdm import tqdm

class Converters:
    def __init__(self) -> None:
        pass
    def to_wave(self, image):
        """this function converts images into k-space wave form"""
        ft = np.fft.fftshift(image)
        ft = np.fft.fft2(ft)
        ft = np.fft.ifftshift(ft)
        return ft
    def normalize(self, data):
        pass