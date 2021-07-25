from .patch_provider import PatchProvider
import faiss
import numpy as np


class FaissANN(PatchProvider):
    """Approximate nearest neighbor search using FAISS
    """

    def get_patch(self, ref_patch):
        return super().get_patch(ref_patch)

    def train(self, patches):
        self._patches = patches
        data = np.array([p.astype(np.float32).ravel() for p in patches])
        self._index = faiss.IndexFlatL2(data.shape[1])
        self._index.add(data)

    def get_patch(self, ref_patch):
        distance, indices = self._index.search(
            np.array([ref_patch.astype(np.float32).ravel()]), k=1)
        return self._patches[indices[0][0]]
