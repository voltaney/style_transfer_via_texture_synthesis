from .patch_provider import PatchProvider
import numpy as np


class NN(PatchProvider):
    """Nearest neighbor search

    This is not really practical due to the execution time.
    Just for testing. 
    """

    def get_patch(self, ref_patch):
        min_distance = np.inf
        result = None
        for i_patch in self._patches:
            distance = np.linalg.norm(ref_patch-i_patch)
            if distance < min_distance:
                min_distance = distance
                result = i_patch
        return result

    def train(self, input_patches):
        self._patches = input_patches
