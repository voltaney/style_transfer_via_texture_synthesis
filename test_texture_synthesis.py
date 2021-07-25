from os import path
import cv2
from src.texture_synthesis import TextureSynthesis


def main():
    filename = 'wara.jpg'
    in_dir = './test_images/texture_synthesis/input_sample/'
    out_dir = './test_images/texture_synthesis/output/'
    input_sample = cv2.imread(path.join(in_dir, filename))
    output_texture = TextureSynthesis(
        input_image=input_sample,
        output_size=(256, 256),
        resolution_layer=3,
        patch_size_list=((32, 32), (16, 16), (8, 8)),
        patch_spacing_list=((8, 8), (4, 4), (2, 2)),
        iteration_n=10,
    ).search_by_HierarchicalNN(
        cluster_num=4,
        patch_amount_tol=100
    ).aggregate_by_lp_irls(
        irls_iteration=10,
        p_norm=1.2,
    ).synthesis()
    cv2.imwrite(path.join(out_dir, filename), output_texture)


if __name__ == '__main__':
    main()
