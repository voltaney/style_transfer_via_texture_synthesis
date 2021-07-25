import cv2
import numpy as np


def sobel_edge_weight(image, ksize=3):
    """Generate weight mat by sobel filter
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y,
                           0.5, 0).astype(np.float64) / 255
    return np.dstack([grad] * image.shape[-1])


def laplacian_edge_weight(image):
    """Generate weight mat by laplacian filter
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)/255
    return np.dstack([lap] * 3)
