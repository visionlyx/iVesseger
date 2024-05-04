from scipy import ndimage
from utils.tiff_read import *
import math

def compare(pre_image, aft_image, dil_image, pixel_value):
    temp_image = aft_image - pre_image
    temp_image = np.where(temp_image == 255, pixel_value, 0)
    dil_image = dil_image + temp_image

    return dil_image

def stop_dilate(aft_image, label, pre_image):
    aft_temp = label - aft_image
    aft_num = np.sum(aft_temp == 255)

    pre_temp = label - pre_image
    pre_num = np.sum(pre_temp == 255)

    return aft_num == pre_num

def expend(skel_image, lab_image):
    pixel_value = 255
    skel_img = np.array(skel_image)
    image = np.array(lab_image)

    dil_image = skel_img.copy()
    expand_img = ndimage.binary_dilation(skel_img, iterations=1, mask=image).astype(skel_img.dtype) * 255
    dil_image = compare(skel_img, expand_img, dil_image, pixel_value)

    expand_img_copy = expand_img.copy()
    temp_img = np.zeros(((dil_image.shape[0], dil_image.shape[1], dil_image.shape[2])))

    while (stop_dilate(expand_img_copy, image, temp_img) == False):
        temp_img = expand_img.copy()
        expand_img = ndimage.binary_dilation(expand_img, iterations=1, mask=image).astype(expand_img.dtype) * 255
        pixel_value = math.floor(pixel_value * 0.96)
        dil_image = compare(expand_img_copy, expand_img, dil_image, pixel_value)
        expand_img_copy = expand_img.copy()

    dil_image = dil_image.astype(np.uint8)
    return dil_image



