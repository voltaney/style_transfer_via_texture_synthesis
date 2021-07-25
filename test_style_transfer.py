from src.style_transfer import StyleTransfer, WeightMatBuilder
import cv2
from os import path


def main():
    base_path = './test_images/style_transfer/'
    content_dir = 'content'
    style_dir = 'style'
    output_dir = 'output'
    style_image = cv2.imread(path.join(base_path, style_dir, 'van_gogh_starry.jpg'))
    content_image = cv2.imread(path.join(base_path, content_dir, 'city_river.jpg'))

    weight_mat = WeightMatBuilder(
        content_image
    ).add_identity(
        weight=0.1,
    ).add_sobel_edge(
        weight=0.9,
        dilation_n=0,
        dilation_kernel=(3, 3)
    ).get_mat()
    output_image = StyleTransfer(
        style_image=style_image,
        content_image=content_image,
        resolution_layer=3,
        patch_size_list=((32, 32), (16, 16), (8, 8)),
        patch_spacing_list=((16, 16), (8, 8), (4, 4)),
        iteration_n=3,
    ).search_by_HierarchicalNN(
        cluster_num=4,
        patch_amount_tol=100,
    ).aggregate_by_lp_irls(
        irls_iteration=10,
        p_norm=1.0,
    ).set_weight_mat(
        weight_mat
    ).transfer()
    cv2.imwrite(path.join(base_path, output_dir, 'result.jpg'), output_image)


if __name__ == '__main__':
    main()
