import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def hist_eq(img, nb):

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    plt.hist(img.flatten(), nb, [0, 256])
    plt.xlim([0, 256])
    plt.legend(('histogram'), loc='upper left')
    plt.show()

    fig2, ax2 = plt.subplots(1, 2)
    equ = cv.equalizeHist(grey)
    ax2[0].imshow(equ)
    cv.imshow("equ", equ)
    hist, bins = np.histogram(equ.flatten(), 256, [0, 256])
    plt.hist(equ.flatten(), nb, [0, 256])
    plt.xlim([0, 256])
    plt.legend(('histogram'), loc='upper left')
    plt.show()

