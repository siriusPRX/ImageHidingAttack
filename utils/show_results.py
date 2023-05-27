import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def array2img(arr):
    '''
    1. convert (c, h, w) into (h, w, c)。
    2. convert the numpy matrix into uint8 format images
    Args:
        arr: numpy matrix

    Returns: uint8 format images

    '''
    arr = np.transpose(arr, (1, 2, 0))
    img = (arr * 255.).astype('uint8')
    img = np.clip(img, 0, 255)
    img = Image.fromarray(img)
    return img

def showResult(show_covers, show_secrets, show_containers, show_revealedSecrets, show_extractedSecrets, show_num, seed=None):
    '''
    display part of the image pairs of cover、secret、container、revealedSecret、extractedSecret.
    Args:
        show_covers:            the matrix of cover images.
        show_secrets:           the matrix of secret images matrix.
        show_containers:        the matrix of container images.
        show_revealedSecrets:   the matrix of revealed secret images.
        show_extractedSecrets:  the matrix of extracted secret images.
        show_num:   Number of the above image pairs to display.
        seed:       Random display.

    Returns:    None

    '''
    np.random.seed(seed)
    idx = np.arange(len(show_containers))
    np.random.shuffle(idx)
    for i in range(show_num):
        cover = array2img(show_covers[idx[i]])
        secret = array2img(show_secrets[idx[i]])
        container = array2img(show_containers[idx[i]])
        revealedSecret = array2img(show_revealedSecrets[idx[i]])
        extractedSecret = array2img(show_extractedSecrets[idx[i]])
        plt.subplot(show_num, 5, i * 5 + 1)
        plt.axis('off')
        if i == 0:
            plt.title('Cover')
        plt.imshow(cover)
        plt.subplot(show_num, 5, i * 5 + 2)
        plt.axis('off')
        if i == 0:
            plt.title('Secret')
        plt.imshow(secret)
        plt.subplot(show_num, 5, i * 5 + 3)
        plt.axis('off')
        if i == 0:
            plt.title('Container')
        plt.imshow(container)
        plt.subplot(show_num, 5, i * 5 + 4)
        plt.axis('off')
        if i == 0:
            plt.title('revealed')
        plt.imshow(revealedSecret)
        plt.subplot(show_num, 5, i * 5 + 5)
        plt.axis('off')
        if i == 0:
            plt.title('extracted')
        plt.imshow(extractedSecret)
    plt.savefig('show.png')