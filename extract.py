import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ImageHidingDataset import ImageHidingDataset
import Model as model

from utils import AverageMeter, reflection_padding, divide_patch, merge_patch
from utils import saveResult, showResult

if __name__ == '__main__':
    test_bz = 1
    num_workers = 4
    model_name = ''
    weight_name = 'epoch_969_0.8701.pth'
    save_name = 'ImageNet/ds/'
    load_dir = './weights/' + model_name + weight_name
    save_dir = './results/' + model_name + save_name
    is_saveResult = True
    print(save_dir)
    
    is_patch = True
    patch_size = (64, 64)
    is_swell = True

    show_covers = []
    show_secrets = []
    show_containers = []
    show_revealedSecrets = []
    show_extractedSecrets = []

    kef = model.UnetPlusPlus(
        'timm-efficientnet-b7',
        classes=3,
        activation=nn.Sigmoid,)

    kef.load_state_dict(torch.load(load_dir, map_location='cpu')['state_dict'])
    kef = kef.cuda()

    test_revLosses = AverageMeter()

    testSet = ImageHidingDataset(root=['./datasets/ImageNet/ds/',
                                       #'./datasets/ImageNet/HiNet/',
                                       #'./datasets/ImageNet/TRM/',
                                       #'./datasets/ImageNet/ISN/',

                                       #'./datasets/COCO/ds/',
                                       # './datasets/COCO/HiNet/',
                                       # './datasets/COCO/TRM/',
                                       # './datasets/COCO/ISN/',

                                       #'./datasets/DIV2K/ds/',
                                       # './datasets/DIV2K/HiNet/',
                                       # './datasets/DIV2K/TRM/',
                                       # './datasets/DIV2K/ISN/',
                                       ],
                                  type='test', testFold_size=20, auto_pad=False, load_cover_secret=True)

    testLoader = DataLoader(dataset=testSet, batch_size=test_bz, num_workers=num_workers, shuffle=False)

    # val
    kef.eval()
    test_process = tqdm(testLoader, mininterval=0)
    cur_val_batch = 0
    test_revLosses.reset()
    with torch.no_grad():
        for idx, (cover, secret, container, revealedSecret) in enumerate(test_process):
            cur_val_batch += len(container)
            if is_patch:
                cover, secret, container, revealedSecret = cover.numpy(), secret.numpy(), container.numpy(), revealedSecret.numpy()
                cover, padding = reflection_padding(cover, patch_size, is_swell=is_swell)
                secret, padding = reflection_padding(secret, patch_size, is_swell=is_swell)
                container, padding = reflection_padding(container, patch_size, is_swell=is_swell)
                revealedSecret, padding = reflection_padding(revealedSecret, patch_size, is_swell=is_swell)
                img_shape, img_dtype = container.shape, container.dtype
                cover, secret, container, revealedSecret = divide_patch(cover, patch_size, is_swell=is_swell), divide_patch(secret, patch_size, is_swell=is_swell), divide_patch(container, patch_size, is_swell=is_swell), divide_patch(revealedSecret, patch_size, is_swell=is_swell)
                cover, secret, container, revealedSecret = torch.from_numpy(cover), torch.from_numpy(secret), torch.from_numpy(container), torch.from_numpy(revealedSecret)
            
            cover, secret, container, revealedSecret = cover.cuda(), secret.cuda(), container.cuda(), revealedSecret.cuda()
            outputs = kef(container)

            test_process.set_description(
                'Test, Batch: {}/{}'.format(
                    cur_val_batch, testSet.__len__())
            )
            cover, secret, container, revealedSecret, outputs = cover.cpu().numpy(), secret.cpu().numpy(), container.cpu().numpy(), revealedSecret.cpu().numpy(), outputs.cpu().numpy()
            if is_patch:
                cover = merge_patch(img_shape, img_dtype, cover, patch_size, is_swell=is_swell, padding=padding)
                secret = merge_patch(img_shape, img_dtype, secret, patch_size, is_swell=is_swell, padding=padding)
                container = merge_patch(img_shape, img_dtype, container, patch_size, is_swell=is_swell, padding=padding)
                revealedSecret = merge_patch(img_shape, img_dtype, revealedSecret, patch_size, is_swell=is_swell, padding=padding)
                outputs = merge_patch(img_shape, img_dtype, outputs, patch_size, is_swell=is_swell, padding=padding)
            show_covers.append(cover[0])
            show_secrets.append(secret[0])
            show_containers.append(container[0])
            show_revealedSecrets.append(revealedSecret[0])
            show_extractedSecrets.append(outputs[0])

    showResult(show_covers, show_secrets, show_containers, show_revealedSecrets, show_extractedSecrets,
               show_num=2, seed=2022)
    if is_saveResult:
        saveResult(show_covers, show_secrets, show_containers, show_revealedSecrets, show_extractedSecrets,
                   save_dir, save_num='all', seed=2022)





