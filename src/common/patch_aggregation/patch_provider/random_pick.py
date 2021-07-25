import random
from .patch_provider import PatchProvider


class RandomPick(PatchProvider):
    def get_patch(self, _):
        return random.choice(self._patches)

    def train(self, input_patches):
        self._patches = input_patches
