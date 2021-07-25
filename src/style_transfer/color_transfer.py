import color_transfer
import skimage.exposure
import numpy as np


def _preprocess(func):
    def wrapper(src, dst):
        original_dtype = src.dtype
        src = src.astype(np.uint8)
        dst = dst.astype(np.uint8)
        result = func(src, dst)
        return result.astype(original_dtype)
    return wrapper


@_preprocess
def lha_color_transfer(src, dst):
    return color_transfer.color_transfer(src, dst)


@_preprocess
def histmatch_color_transfer(src, dst):
    return skimage.exposure.match_histograms(dst, src, multichannel=True)
