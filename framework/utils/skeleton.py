from skimage.morphology import skeletonize_3d

def extract_centerlines(segmentation):
    skeleton = skeletonize_3d(segmentation)
    skeleton.astype(dtype='uint8', copy=False)
    return skeleton



