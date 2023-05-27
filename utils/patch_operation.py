import numpy as np

def divide_patch(img, patch_size=(64, 64), is_swell=True):
    '''

    Args:
        img:
        patch_size: default is (64, 64).
        is_swell: default is True. Take a patch at 1/2 patch_size intervals.

    Returns: patches of image.

    '''
    imgs = []
    h, w = img.shape[-2:]
    if not is_swell:
        stride = patch_size
    else:
        stride = (patch_size[0]//2, patch_size[1]//2)
    for i in range(0, h, stride[0]):
        for j in range(0, w, stride[1]):
            if i + patch_size[0] <= h and j + patch_size[1] <= w:
                a = img[:, :, i:i + patch_size[0], j:j + patch_size[1]]
                imgs.append(a)
    imgs = np.vstack(imgs)
    return imgs

def merge_patch(img_shape, img_dtype, imgs, patch_size, is_swell=False, padding=(0, 0, 0, 0)):
    b, c, h, w = img_shape
    if not is_swell:
        col, row = h // patch_size[0], w // patch_size[1]
    else:
        col, row = (h // patch_size[0]) * 2, (w // patch_size[1]) * 2
    img = np.zeros(img_shape, dtype=img_dtype)
    count=0
    for i in range(row):
        for j in range(col):
            for k in range(b):
                if not is_swell:
                    top = i * patch_size[0]
                    down = top + patch_size[0]
                    left = j * patch_size[1]
                    right = left + patch_size[1]
                    img[k, :, top:down, left:right] = imgs[count]
                    count += 1
                else:
                    top = patch_size[0]//4 + i * patch_size[0]//2
                    down = top + patch_size[0]//2
                    left = patch_size[1]//4 + j * patch_size[1]//2
                    right = left + patch_size[1]//2
                    img[k, :, top:down, left:right] = imgs[count, :, patch_size[0]//4:patch_size[0]*3//4, patch_size[1]//4:patch_size[1]*3//4]
                    count += 1
    return img[:, :, padding[0]:h-padding[1], padding[2]:w-padding[3]]