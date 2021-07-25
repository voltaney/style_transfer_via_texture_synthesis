from sklearn.cluster import KMeans
import numpy as np
from .patch_provider import PatchProvider


class HierarchicalNN(PatchProvider):
    """Nearest neighbor search using hierarchical clustering
    """

    def __init__(self, n_clusters=4, patch_amount_tol=100):
        self.n_clusters = n_clusters
        self.patch_amount_tol = patch_amount_tol
        self.kmeans = KMeans(self.n_clusters)
        self.children = []
        self.candidate_patches = None

    def train(self, patches):
        self.candidate_patches = patches
        if len(self.candidate_patches) <= self.patch_amount_tol:
            return
        dataset = [img.ravel() for img in patches]
        self.kmeans.fit(dataset)
        labels = self.kmeans.predict(dataset)
        for i in range(self.n_clusters):
            subset_search = HierarchicalNN(
                n_clusters=self.n_clusters,
                patch_amount_tol=self.patch_amount_tol)
            subset_patches = []
            for idx, label in enumerate(labels):
                if label == i:
                    subset_patches.append(self.candidate_patches[idx])
            subset_search.train(subset_patches)
            self.children.append(subset_search)

    def get_patch(self, ref_patch):
        return self.search(ref_patch)

    def get_patch_provider(self):
        def __internal(patch):
            return self.search(patch)
        return __internal

    def search(self, patch):
        if not self.children:
            return self.__nearest_neighborhood(patch)
        label = self.kmeans.predict([patch.ravel()])[0]
        return self.children[label].search(patch)

    def __nearest_neighborhood(self, patch):
        min_distance = np.inf
        result = None
        if self.candidate_patches is None:
            raise LookupError('unable to search neighborhoods')
        for c_patch in self.candidate_patches:
            distance = np.linalg.norm(patch-c_patch)
            if distance < min_distance:
                min_distance = distance
                result = c_patch
        return result
