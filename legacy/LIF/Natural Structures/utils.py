import numpy as np
from sklearn.preprocessing import minmax_scale
import cv2
import math
import pandas as pd
from tqdm import tqdm

def to_current(img: np.ndarray):
    img = img.astype(np.float64)
    img *= 255 / np.max(img)
    img_as_current = minmax_scale(img, feature_range=(-80, -40))
    return img_as_current

def to_vector(img: np.ndarray):
    return img.flatten(order="C")

def pad_image(img: np.ndarray, k_size: int):
    if len(img.shape) > 2:
        img = np.reshape(img(img.shape[0], img.shape[1]))
    elif len(img.shape) < 2:
        raise Exception("Image is 1D!")

    img_h, img_w = img.shape
    h_pad = 0
    w_pad = 0
    h_gcd = np.gcd(img_h, k_size)
    w_gcd = np.gcd(img_w, k_size)

    if h_gcd < k_size:
        h_pad += k_size - h_gcd
    if w_gcd < k_size:
        w_pad += k_size - w_gcd

    padded_img = np.pad(img, ((h_pad, w_pad), (h_pad, w_pad)), mode="edge")
    return padded_img

def test_pad():
    img = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)
    img = pad_image(img, 3)
    cv2.imshow("padded", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    raise

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
