import abc


class PatchProvider(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_patch(self, ref_patch):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, input_patches):
        raise NotImplementedError
