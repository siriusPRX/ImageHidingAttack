import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from .averagemeter import AverageMeter

def transArray(arr):
    '''
    1. convert (c, h, w) into (h, w, c)。
    2. convert the numpy matrix into uint8 format images.
    Args:
        arr: numpy matrix

    Returns: uint8 format numpy matrix

    '''
    arr = np.transpose(arr, (1, 2, 0))
    img = (arr * 255.).astype('uint8')
    img = np.clip(img, 0, 255)
    return img

def saveResult(show_covers, show_secrets, show_containers, show_revealedSecrets, show_extractedSecrets,
               save_dir, save_num='all', seed=None):
    '''
    save image pairs of cover、secret、container、revealedSecret、extractedSecret.
    Args:
        show_covers:            the matrix of cover images.
        show_secrets:           the matrix of secret images matrix.
        show_containers:        the matrix of container images.
        show_revealedSecrets:   the matrix of revealed secret images.
        show_extractedSecrets:  the matrix of extracted secret images.
        save_dir:   The root saving directory.
        save_num:   Number of images to save, default is ALL.
        seed:       Random if save_num not all.

    Returns:

    '''
    # create saving subdir for different images.
    cover_dir = os.path.join(save_dir, 'covers/')
    if not os.path.exists(cover_dir):
        os.makedirs(cover_dir)
    secret_dir = os.path.join(save_dir, 'secrets/')
    if not os.path.exists(secret_dir):
        os.makedirs(secret_dir)
    container_dir = os.path.join(save_dir, 'containers/')
    if not os.path.exists(container_dir):
        os.makedirs(container_dir)
    predict_dir = os.path.join(save_dir, 'extractedSecrets/')
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
    revealedSecret_dir = os.path.join(save_dir, 'revealedSecrets/')
    if not os.path.exists(revealedSecret_dir):
        os.makedirs(revealedSecret_dir)

    np.random.seed(seed)
    idx = np.arange(len(show_containers))
    np.random.shuffle(idx)

    if save_num == 'all':
        save_num = len(show_revealedSecrets)
    save_progress = tqdm(range(save_num))
    # psnr and ssim value between secret image and revealed secret image.
    psnr_sp, ssim_sp = AverageMeter(), AverageMeter()
    # psnr and ssim value between secret image and extracted secret image.
    psnr_se, ssim_se = AverageMeter(), AverageMeter()

    for i in save_progress:
        cover = transArray(show_covers[idx[i]])
        secret = transArray(show_secrets[idx[i]])
        container = transArray(show_containers[idx[i]])
        revealSecret = transArray(show_revealedSecrets[idx[i]])
        extractedSecret = transArray(show_extractedSecrets[idx[i]])

        psnr_sp.update(PSNR(secret, revealSecret))
        ssim_sp.update(SSIM(secret, revealSecret, channel_axis=2))
        psnr_se.update(PSNR(secret, extractedSecret))
        ssim_se.update(SSIM(secret, extractedSecret, channel_axis=2))

        cover = Image.fromarray(cover)
        secret = Image.fromarray(secret)
        container = Image.fromarray(container)
        revealSecret = Image.fromarray(revealSecret)
        extractedSecret = Image.fromarray(extractedSecret)

        cover.save(os.path.join(cover_dir, 'cover_{}.png'.format(str(i).zfill(5))))
        secret.save(os.path.join(secret_dir, 'secret_{}.png'.format(str(i).zfill(5))))
        container.save(os.path.join(container_dir, 'container_{}.png'.format(str(i).zfill(5))))
        revealSecret.save(os.path.join(revealedSecret_dir, 'revealedSecret_{}.png'.format(str(i).zfill(5))))
        extractedSecret.save(os.path.join(predict_dir, 'extractedSecret_{}.png'.format(str(i).zfill(5))))

        save_progress.set_description(
            'PSNR_sp:{} | SSIM_sp:{} | PSNR_se:{} | SSIM_se:{}'.format(psnr_sp.avg, ssim_sp.avg,
                                                                       psnr_se.avg, ssim_se.avg))