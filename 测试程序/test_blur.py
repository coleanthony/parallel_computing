import sys
import cv2
import numpy as np
from math import sqrt

def compare_mse(pic1, pic2):
    diff = np.sum((pic1.astype("float") - pic2.astype("float")) ** 2)
    return diff / float(pic1.shape[0] * pic1.shape[1] * pic1.shape[2])

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print("Usage: python3 test_blur.py [teacher's image] [your image]")
        exit()
    
    img = cv2.imread("Pure_black.bmp")
    test = cv2.imread("Pure_white.bmp")

    kernel = np.array([
        [0.01441881, 0.02808402, 0.03507270, 0.02808402, 0.01441881],
        [0.02808402, 0.05470020, 0.06831229, 0.05470020, 0.02808402],
        [0.03507270, 0.06831229, 0.08531173, 0.06831229, 0.03507270],
        [0.02808402, 0.05470020, 0.06831229, 0.05470020, 0.02808402],
        [0.01441881, 0.02808402, 0.03507270, 0.02808402, 0.01441881]
    ])

    blur = cv2.filter2D(src=img, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)

    if (blur.shape != test.shape):
        print("[Fail] Image shape mismatch: should be {0}, yours {1}".format(blur.shape, test.shape))
        exit()
    else:
        print("[Good] Image shape matched.")

    mse = compare_mse(blur, test)
    sim = 100 - sqrt(mse)/2.55
    print("MSE result of the two pictures: {:.2f}".format(mse))
    print("Similarity rate: {:.2f}%".format(sim))