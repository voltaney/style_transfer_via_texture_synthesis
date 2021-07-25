import numpy as np


def no_segmentation(image):
    return np.ones(image.shape, dtype=np.float64)
