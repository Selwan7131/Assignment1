import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from math import *

def filter_img(img):
    mean = 0
    var = 0.005
    # sigma = var ** 0.5
    gaussian = np.random.normal(mean, var, (360, 520))

    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    cv.normalize(noisy_image, noisy_image, 0, 255, cv.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)

    cv.imshow("img", img)
    cv.imshow("gaussian", gaussian)
    cv.imshow("noisy", noisy_image)
    cv.waitKey(0)

def box_filter(img):
    size = 3
    box_filter_img = cv.filter2D(img, -1, box_kernel(size))
    cv.imshow("box filter", box_filter_img)
    cv.waitKey(0)

def box_filter_return(img):
    size = 3
    box_filter_img = cv.filter2D(img, -1, box_kernel(size))
    return box_filter_img

def box_kernel(size):
  k = np.ones((size,size),np.float32)/(size**2)
  return k

def median_filter(img):
    median = cv.medianBlur(img, 5)
    compare = np.concatenate((img, median), axis=1)
    cv.imshow('img', compare)
    cv.waitKey(0)
    cv.destroyAllWindows

def median_filter_return(img):
    median = cv.medianBlur(img, 5)
    return median

def mse(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    return mse

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
