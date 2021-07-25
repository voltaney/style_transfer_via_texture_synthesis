import numpy as np

from .extract_patches import extract_patches

# avoid zero division
NOISE = 0.000001


def l2_norm_aggregate(initial_output_image, patch_provider, patch_size, patch_spacing):
    """Aggregate by L2 norm optimization

    Args:
        initial_output_image (ndarray): Target image to be aggregated
        patch_provider (PatchProvider): Patch provider
        patch_size (int, int): Patch size(width, height)
        patch_spacing (int, int): Patch sampling gap(width, height)

    Returns:
        ndarray: Aggregated result
    """
    new_output_image = np.zeros(initial_output_image.shape, dtype=np.float64)
    addition_count_mat = np.zeros(initial_output_image.shape, dtype=np.float64)
    new_output_patches = extract_patches(new_output_image, patch_size, patch_spacing)
    addition_count_patches = extract_patches(addition_count_mat, patch_size, patch_spacing)
    initial_output_patches = extract_patches(initial_output_image, patch_size, patch_spacing)

    for i in range(len(new_output_patches)):
        new_output_patches[i][:] += patch_provider.get_patch(initial_output_patches[i])
        addition_count_patches[i][:] += 1
    addition_count_mat[addition_count_mat == 0] = 1
    new_output_image[:] /= addition_count_mat
    return new_output_image


def lp_norm_irls_aggregate(initial_output_image, patch_provider, patch_size, patch_spacing, irls_iteration=10, p_norm=1.2, irls_tol=0):
    """Aggregate by IRLS robust optimization (LP norm linear regression)

    Args:
        initial_output_image (ndarray): Target image to be aggregated
        patch_provider (PatchProvider): Patch provider
        patch_size (int, int): Patch size(width, height)
        patch_spacing (int, int): Patch sampling gap(width, height)
        irls_iteration (int, optional): Number of iteration for IRLS
        p_norm (float, optional): P-norm for IRLS
        irls_tol (float, optional): Convergence condition for IRLS (Rate of change of mean deviation)

    Returns:
        ndarray: Aggregated result
    """
    aggregate_result_image = initial_output_image.copy().astype(np.float64)

    aggregate_result_patches = extract_patches(aggregate_result_image, patch_size, patch_spacing)

    weight_mat = np.zeros(initial_output_image.shape, dtype=np.float64)
    weight_patches = extract_patches(weight_mat, patch_size, patch_spacing)
    initial_output_patches = extract_patches(initial_output_image, patch_size, patch_spacing)

    match_result_patches = []
    for p in initial_output_patches:
        match_result_patches.append(patch_provider.get_patch(p))
    pre_distance_sum = 0
    for itr in range(irls_iteration):
        weight_mat[:] = 0
        weights = []
        distance_sum = 0
        for i in range(len(aggregate_result_patches)):
            distance = np.linalg.norm(match_result_patches[i] - aggregate_result_patches[i])
            distance += NOISE
            weights.append(np.power(distance, p_norm-2))
            distance_sum += distance
        # check convergence condition
        if irls_tol and itr != 0 and np.abs(distance_sum-pre_distance_sum)/pre_distance_sum < irls_tol:
            break
        pre_distance_sum = distance_sum
        aggregate_result_image[:] = 0
        for i in range(len(aggregate_result_patches)):
            weight_patches[i][:] += weights[i]
            aggregate_result_patches[i][:] += match_result_patches[i]*weights[i]
        aggregate_result_image[:] /= (weight_mat+NOISE)
    return aggregate_result_image
