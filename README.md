# Paper Implementation
The following two papers have been implemented.

Some of the implementations are incomplete or have additional methods that are not in the papers.

- V. Kwatra, I. Essa, A. Bobick and N. Kwatra, "Texture optimization for example-based synthesis", ACM Trans. Graph., vol. 24, no. 3, pp. 795-802, 2005.
- M. Elad and P. Milanfar, "Style Transfer Via Texture Synthesis," in IEEE Transactions on Image Processing, vol. 26, no. 5, pp. 2338-2351, May 2017

## Texture optimization for example-based synthesis
| Input sample | Output |
----| ---- 
|![input_sample](test_images/texture_synthesis/input_sample/wara.jpg "input_sample") | ![output](test_images/texture_synthesis/output/wara.jpg "output")|

### How to use
See or run `test_texture_synthesis.py`.

### Not implemented method (propsed in the paper)
- Call-off function applied to the patch during the EM algorithm in the paper.

## Style Transfer Via Texture Synthesis
| Content | Style | Result |
----| ---- | ----
| <img src="test_images/style_transfer/content/city_river.jpg" width="200"/> | <img src="test_images/style_transfer/style/van_gogh_starry.jpg" width="200"/> | <img src="test_images/style_transfer/output/result.jpg" width="200"/>|

### How to use
See or run `test_style_transfer.py`.

### Not implemented method (propsed in the paper)
- Any segmentation
    - Instead, using edge detection to calculate the weights and fuse contents.
- Denoise by domain-transform
- ANN using PCA