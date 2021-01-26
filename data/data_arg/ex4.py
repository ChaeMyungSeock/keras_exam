import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Initialize the bounding box for the original image
# using helpers from imgaug package
bbs = BoundingBoxesOnImage([
    BoundingBox(x1=1, x2=980, y1=9, y2=535)
], shape=image.shape)

# Define a simple augmentations pipeline for the image with bounding box
seq = iaa.Sequential([
    iaa.GammaContrast(1.5), # add contrast
    iaa.Affine(translate_percent={"x": 0.1}, scale=0.8), # translate the image
    iaa.Fliplr(p = 1.0) # apply horizontal flip
])

# Apply augmentations
image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)