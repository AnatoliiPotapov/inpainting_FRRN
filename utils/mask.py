import numpy as np


def get_mask(img):
    """ Selects (255, 255, 255) pixels as mask """
    # TODO Improve this
    pixel_intensity = np.sum(img, axis=2, keepdims=True)
    return pixel_intensity == 765