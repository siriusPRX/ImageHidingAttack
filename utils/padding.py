import numpy as np

def reflection_padding(img, patch_size, is_swell=False):
    '''

    Args:
        img:
        patch_size:
        is_swell:

    Returns:

    '''
    h, w = img.shape[-2:]

    h_pad = (patch_size[0] - h % patch_size[0]) % patch_size[0]
    w_pad = (patch_size[1] - w % patch_size[1]) % patch_size[1]

    top_pad = h_pad // 2
    down_pad = h_pad - top_pad
    left_pad = w_pad // 2
    right_pad = w_pad - left_pad
    img = np.pad(img, ((0, 0), (0, 0), (top_pad, down_pad), (left_pad, right_pad)), 'reflect')
    if not is_swell:
        return img, (top_pad, down_pad, left_pad, right_pad)
    img = np.pad(img,
                 ((0, 0), (0, 0), (patch_size[0] // 4, patch_size[0] // 4), (patch_size[1] // 4, patch_size[1] // 4)),
                 'reflect')
    return img, (top_pad + patch_size[0] // 4, down_pad + patch_size[0] // 4, left_pad + patch_size[1] // 4,
                 right_pad + patch_size[1] // 4)