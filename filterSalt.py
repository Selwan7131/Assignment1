import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from equ import *
from filter import *


def s_and_p_noise(image, s_p_ratio=0.05):
    out = np.copy(image)

    # Salt mode
    mask = np.random.rand(image.shape[0], image.shape[1]) <= s_p_ratio / 2
    out[mask] = 255

    # Pepper mode
    mask = np.random.rand(image.shape[0], image.shape[1]) <= s_p_ratio / 2
    out[mask] = 0
    return out
