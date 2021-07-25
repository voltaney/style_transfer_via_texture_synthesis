import numpy as np
import cv2
from numpy.core.fromnumeric import _resize_dispatcher
from ..common.pyramid import get_pyramid
from ..common.patch_aggregation import extract_patches, l2_norm_aggregate, lp_norm_irls_aggregate
from ..common.patch_aggregation.patch_provider import RandomPick, HierarchicalNN, FaissANN, NN
from tqdm import tqdm


class TextureSynthesis:
    def __init__(self, input_image, output_size, resolution_layer, patch_size_list, patch_spacing_list, iteration_n):
        """Texture synthesis

        Args:
            input_image (ndarray): Input image
            output_size (tuple): Size of the image to be synthesized
            resolution_layer (int): Number of resolution layer
            patch_size_list (tuple): List of patch sizes
            patch_spacing_list (tuple): List of patch sampling gaps
            iteration_n (int): Number of iteration
        """
        if resolution_layer != len(patch_size_list) or resolution_layer != len(patch_spacing_list):
            raise ValueError('Invalid patch settings')
        self.input_image = input_image
        self.output_size = output_size
        self.resolution_layer = resolution_layer
        self.patch_size_list = patch_size_list
        self.patch_spacing_list = patch_spacing_list
        self.iteration_n = iteration_n
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

    def synthesis(self):
        if not self.patch_aggregator:
            raise ValueError('Patch aggregator is not set')
        if not self.patch_provider_builder:
            raise ValueError('Patch provider is not set')
        # prepare pyramid
        input_pyramid = get_pyramid(self.input_image, self.resolution_layer)

        # initialze
        output_shape = list(self.output_size)
        if len(self.input_image.shape) == 3:
            output_shape.append(self.input_image.shape[2])
        output_texture = get_pyramid(np.zeros(output_shape, dtype=np.float64), self.resolution_layer)[0]
        input_patches_for_init = extract_patches(input_pyramid[0], self.patch_size_list[-1], self.patch_spacing_list[-1])
        rp = RandomPick()
        rp.train(input_patches_for_init)
        output_texture = self.patch_aggregator(
            output_texture,
            rp,
            self.patch_size_list[-1],
            self.patch_spacing_list[-1]
        )

        # synthesis
        for r_idx in range(self.resolution_layer):
            resized_input_image = input_pyramid[r_idx]
            for patch_size, patch_spacing in zip(self.patch_size_list[-1-r_idx:], self.patch_spacing_list[-1-r_idx:]):
                input_patches = extract_patches(resized_input_image, patch_size, patch_spacing)
                patch_provider = self.patch_provider_builder()
                patch_provider.train(input_patches)
                print(f'Layer: {output_texture.shape}, Patch: {patch_size}')
                for i in tqdm(range(self.iteration_n)):
                    output_texture = self.patch_aggregator(output_texture, patch_provider, patch_size, patch_spacing)

            if r_idx < self.resolution_layer-1:
                output_texture = cv2.pyrUp(output_texture.astype(np.uint8)).astype(np.float64)

        return output_texture.astype(np.uint8)
