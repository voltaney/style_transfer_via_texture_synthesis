def extract_patches(input_img, patch_size, patch_spacing):
    """Patch extraction

    The returned patch have pointers to the input_img.

    Args:
        input_img (ndarray): image
        patch_size (int, int): patch size (width, height)
        patch_spacing (int, int): patch sampling gap (width, height)

    Returns:
        list of ndarray: list of patches.
    """
    patches = []
    p_h, p_w = patch_size
    s_h, s_w = patch_spacing
    max_h, max_w = input_img.shape[:2]
    idx_h, idx_w = (0, 0)
    while 1:
        if idx_h + p_h > max_h or idx_w + p_w > max_w:
            break
        patches.append(input_img[idx_h:idx_h+p_h, idx_w:idx_w+p_w])
        idx_w += s_w
        if idx_w+p_w > max_w:
            idx_w = 0
            idx_h += s_h
    return patches
