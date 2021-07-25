import cv2
import numpy as np
from .content_fusion import no_segmentation, laplacian_edge_weight, sobel_edge_weight


class WeightMatBuilder:
    def __init__(self, content_image):
        self.content_image = content_image.copy()
        self.weight_mat = np.zeros(content_image.shape, dtype=np.float64)

    def add_identity(self, weight=1.0):
        self.weight_mat[:] += weight*no_segmentation(self.content_image)
        return self

    def add_laplacian_edge(self, dilation_n=0, dilation_kernel=(3, 3), weight=1.0):
        mat = laplacian_edge_weight(self.content_image)
        if dilation_n > 0:
            mat = cv2.dilate(mat, dilation_kernel, iterations=dilation_n)
        self.weight_mat[:] += weight * mat
        return self

    def add_sobel_edge(self, dilation_n=0, dilation_kernel=(3, 3), weight=1.0):
        mat = sobel_edge_weight(self.content_image)
        kernel = np.ones(dilation_kernel, dtype=np.uint8)
        if dilation_n > 0:
            mat = cv2.dilate(mat, kernel, iterations=dilation_n)
        self.weight_mat[:] += weight * mat
        return self

    def get_mat(self):
        return self.weight_mat.copy()
