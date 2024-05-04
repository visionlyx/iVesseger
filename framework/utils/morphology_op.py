from utils.image_filter import *
from utils.skeleton import *
from utils.expand import *

def find_max_neighbor(image, regions, z, y, x):
    max_value = 0
    max_point = []

    # Traverse the adjacent points
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nz, ny, nx = z + dz, y + dy, x + dx
                if (0 <= nz < image.shape[0] and 0 <= ny < image.shape[1] and 0 <= nx < image.shape[2]):
                    point = [nz, ny, nx]
                    if ((image[nz, ny, nx] > max_value) and (point not in regions)):
                        max_value = image[nz, ny, nx]
                        max_point = point

    return max_value, max_point


def fp_image_improve(image, z, y, x):
    gauss_image = ndimage.gaussian_filter(image, sigma=1)

    regions = list()
    regions.append([z, y, x])

    mask = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)

    while (len(regions) != 16):
        max_value = 0
        max_point = []
        for point in regions:
            temp_value, temp_point = find_max_neighbor(gauss_image, regions, point[0], point[1], point[2])
            if (temp_value > max_value):
                max_value = temp_value
                max_point = temp_point

        regions.append(max_point)

    for point in regions:
        mask[point[0], point[1], point[2]] = random.randint(1, 5)

    mask = np.array(mask, dtype=np.uint16)
    impro_image = np.where(mask != 0, mask, image)

    return impro_image


def fn_image_improve(image, max_grey, z, y, x):
    gauss_image = ndimage.gaussian_filter(image, sigma=1)

    regions = list()
    regions.append([z, y, x])

    mask = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)

    while (len(regions) != 48):
        max_value = 0
        max_point = []
        for point in regions:
            temp_value, temp_point = find_max_neighbor(gauss_image, regions, point[0], point[1], point[2])
            if (temp_value >= max_value):
                max_value = temp_value
                max_point = temp_point

        regions.append(max_point)

    for point in regions:
        mask[point[0], point[1], point[2]] = 255

    skeleton = extract_centerlines(mask)
    dilate = expend(skeleton, mask)

    dilate = np.array(dilate, dtype=np.float32)
    mutil_value = max_grey / 255
    dilate = dilate * mutil_value

    dilate = np.array(dilate, dtype=np.uint16)
    impro_image = np.where(dilate != 0, dilate, image)

    return impro_image


def morphology_op(image, fp_point_volume, fn_point_volume, image_size):
    fp_crop_size = 10
    fn_crop_size = 10

    max_grey = image.max()

    if len(fp_point_volume) != 0:
        for fp_points in fp_point_volume:
            x, y, z = fp_points

            zb = z - fp_crop_size
            zend = z + fp_crop_size
            yb = y - fp_crop_size
            yend = y + fp_crop_size
            xb = x - fp_crop_size
            xend = x + fp_crop_size

            if (zb < 0):
                zb = 0
            if (yb < 0):
                yb = 0
            if (xb < 0):
                xb = 0

            if (zend > image_size):
                zend = image_size
            if (yend > image_size):
                yend = image_size
            if (xend > image_size):
                xend = image_size

            temp_image = image[zb:zend, yb:yend, xb:xend]
            impro_image = fp_image_improve(temp_image, z-zb, y-yb, x-xb)
            image[zb:zend, yb:yend, xb:xend] = impro_image

    if len(fn_point_volume) != 0:
        for fn_points in fn_point_volume:
            x, y, z = fn_points

            zb = z - fn_crop_size
            zend = z + fn_crop_size
            yb = y - fn_crop_size
            yend = y + fn_crop_size
            xb = x - fn_crop_size
            xend = x + fn_crop_size

            if (zb < 0):
                zb = 0
            if (yb < 0):
                yb = 0
            if (xb < 0):
                xb = 0

            if (zend > image_size):
                zend = image_size
            if (yend > image_size):
                yend = image_size
            if (xend > image_size):
                xend = image_size

            temp_image = image[zb:zend, yb:yend, xb:xend]
            impro_image = fn_image_improve(temp_image, max_grey, z-zb, y-yb, x-xb)
            image[zb:zend, yb:yend, xb:xend] = impro_image

    return image

