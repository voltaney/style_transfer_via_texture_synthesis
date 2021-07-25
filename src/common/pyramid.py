import cv2


def get_pyramid(original_image, size, ASC=True):
    """Generate gaussian pyramid

    Args:
        original_image (ndarray): Original image
        size (int): Size of pyramid
        ASC (bool, optional): Ascending order. Defaults to True.

    Returns:
        list of ndarray: Gaussian pyramid
    """
    pyramid = [original_image]
    for i in range(size-1):
        pyramid.append(cv2.pyrDown(pyramid[i]))
    if ASC:
        pyramid = pyramid[::-1]
    return pyramid
