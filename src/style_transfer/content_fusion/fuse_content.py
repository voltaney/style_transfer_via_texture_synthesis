def fuse_content(target_image, content_image, weight_mat):
    """Fuse content according to weight mat

    Args:
        target_image (ndarray): Target image
        content_image (ndarray): Content image
        weight_mat (ndarray): Weight mat

    Returns:
        ndarray: Fused target image
    """
    return (target_image + content_image*weight_mat)/(1+weight_mat)
