import numpy as np
import cv2
from .color_transfer import histmatch_color_transfer
from ..common.pyramid import get_pyramid
from ..common.patch_aggregation import extract_patches, l2_norm_aggregate, lp_norm_irls_aggregate
from ..common.patch_aggregation.patch_provider import HierarchicalNN, FaissANN, NN
from .content_fusion import fuse_content
from tqdm import tqdm


class StyleTransfer:
    def __init__(self, content_image, style_image, resolution_layer, patch_size_list, patch_spacing_list, iteration_n, init_noise_sigma=50):
        if len(patch_size_list) != len(patch_spacing_list):
            raise ValueError('Invalid patch settings')
        self.content_image = content_image
        self.style_image = style_image
        self.resolution_layer = resolution_layer
        self.patch_size_list = patch_size_list
        self.patch_spacing_list = patch_spacing_list
        self.iteration_n = iteration_n
        self.init_noise_sigma = init_noise_sigma
        self.weight_mat = np.zeros(content_image.shape, dtype=np.float64)
        self.patch_aggregator = None
        self.patch_provider_builder = None

    def search_by_HierarchicalNN(self,  cluster_num, patch_amount_tol):
        """Search NN patches by hierarchical clustering

        Args:
            cluster_num (int): The number of clusters a hierarchy has
            patch_amount_tol (int): Number of patches allowed in a cluster
        """
        def wrapper():
            return HierarchicalNN(cluster_num, patch_amount_tol)
        self.patch_provider_builder = wrapper
        return self

    def search_by_FaissANN(self):
        """Search NN patches by FAISS ANN
        """
        def wrapper():
            return FaissANN()
        self.patch_provider_builder = wrapper
        return self

    def search_by_NN(self):
        """Search NN patches by simple NN (Not practical)
        """
        def wrapper():
            return NN()
        self.patch_provider_builder = wrapper
        return self

    def aggregate_by_l2(self):
        """Aggregate by L2 optimization
        """
        self.patch_aggregator = l2_norm_aggregate
        return self

    def aggregate_by_lp_irls(self, irls_iteration=10, p_norm=1.2, irls_tol=0):
        """Aggregate by IRLS optimization (LP norm linear regression)

        Args:
            irls_iteration (int, optional): Number of iteration for IRLS. Defaults to 10.
            p_norm (float, optional): p-norm. Defaults to 1.2.
            irls_tol (float, optional): Convergence condition for IRLS. Defaults to 0.
        """
        def wrapper(img, provider, size, spacing):
            return lp_norm_irls_aggregate(img, provider, size, spacing, irls_iteration, p_norm, irls_tol)
        self.patch_aggregator = wrapper
        return self

    def set_weight_mat(self, weight_mat):
        if self.content_image.shape != weight_mat.shape:
            raise ValueError(
                'A shape of the weight mat must be same with the content image.')
        self.weight_mat = weight_mat
        return self

    def transfer(self):
        if not self.patch_aggregator:
            raise ValueError('Call aggregare_by_* method before trasnfer()')
        if not self.patch_provider_builder:
            raise ValueError('Call search_by_* method before transfer()')

        color_transfered = histmatch_color_transfer(self.style_image, self.content_image)
        color_transfered = color_transfered.astype(np.float64)
        style_image = self.style_image.astype(np.float64)

        # prepare pyramid
        style_pyramid = get_pyramid(style_image, self.resolution_layer)
        color_transfered_pyramid = get_pyramid(color_transfered, self.resolution_layer)
        weight_mat_pyramid = get_pyramid(self.weight_mat, self.resolution_layer)

        # initialize
        output_image = color_transfered_pyramid[0]+np.random.normal(
            scale=self.init_noise_sigma, size=color_transfered_pyramid[0].shape)

        # style_transfer
        for r_idx in range(self.resolution_layer):
            for patch_size, patch_spacing in zip(self.patch_size_list, self.patch_spacing_list):
                style_patches = extract_patches(style_pyramid[r_idx], patch_size, patch_spacing)
                patch_provider = self.patch_provider_builder()
                patch_provider.train(style_patches)
                print(f'Layer: {output_image.shape}, Patch: {patch_size}')
                for _ in tqdm(range(self.iteration_n)):
                    output_image = self.patch_aggregator(output_image, patch_provider, patch_size, patch_spacing)
                    output_image = fuse_content(output_image, color_transfered_pyramid[r_idx], weight_mat_pyramid[r_idx])
                    output_image = histmatch_color_transfer(style_pyramid[r_idx], output_image)

            if r_idx < self.resolution_layer-1:
                output_image = cv2.pyrUp(output_image)
                # adjust size
                if output_image.shape[0] > color_transfered_pyramid[r_idx+1].shape[0]:
                    output_image = output_image[:color_transfered_pyramid[r_idx+1].shape[0], :]
                if output_image.shape[1] > color_transfered_pyramid[r_idx+1].shape[1]:
                    output_image = output_image[:, :color_transfered_pyramid[r_idx+1].shape[1]]

        return output_image.astype(np.uint8)
