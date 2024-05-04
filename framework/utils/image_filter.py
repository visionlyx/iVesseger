import random
from utils.tiff_read import *

def random_clip(img, percentage1, percentage2):
    rand_per = round(random.uniform(percentage1, percentage2), 7)
    img_flat = img.flatten()
    img_flat = abs(np.sort(img_flat))
    thre_pos = int(np.floor(len(img_flat) * (1 - rand_per)))
    thre_value = img_flat[thre_pos]
    img = np.where(img > thre_value, thre_value, img)
    return img