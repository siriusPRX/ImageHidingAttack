import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from ImageHidingDataset import ImageHidingDataset
import Model as model
import weighed_loss
from utils import AverageMeter, Logger


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    # config
    train_mulGPU = False
    auto_pad = True
    patch_factor = 4
    train_bz = 64
    val_bz = 1
    cur_epoch = 0
    max_epoch = 1000
    num_workers = 8
    model_name = 'kef'
    log_file = './logs/' + model_name
    save_dir = './weights/' + model_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    is_loadWeight = False
    weigth_file = save_dir + '/last_epoch.pth'
    min_val_loss = np.inf
    
    # ---- create model ---- #
    kef = model.UnetPlusPlus(
            'timm-efficientnet-b7',
            classes=3,
            encoder_weights='imagenet',
            activation=nn.Sigmoid,
        )

    # ---- create optim ---- #
    crit = weighed_loss.WeightedLoss([weighed_loss.VGGLoss(shift=2),
                                  nn.MSELoss(),
                                  weighed_loss.TVLoss(p=2)],
                                 [1, 40, 10]).cuda()
    optimizer = optim.AdamW(kef.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=10, verbose=True)
    
    if is_loadWeight:
        kef.load_state_dict(torch.load(weigth_file, map_location='cpu')['state_dict'])
        #optimizer.load_state_dict(torch.load(weigth_file, map_location='cpu')['optimizer'])
        cur_epoch = torch.load(weigth_file, map_location='cpu')['epoch']
    if train_mulGPU and torch.cuda.device_count() > 1:
        kef = torch.nn.DataParallel(kef).cuda()
    else:
        kef = kef.cuda()

    # ----define dataset---- #
    trainSet = ImageHidingDataset(root=['./datasets/ImageNet/ds',
                                        './datasets/ImageNet/HiNet',
                                        './datasets/ImageNet/TRM',
                                        './datasets/ImageNet/ISN',
                                        ],
                                  type='train', trainFold_size=5, valFold_size=1, is_aug=False, auto_pad=auto_pad,
                                  patch_factor=patch_factor, load_cover_secret=False)
    valSet = ImageHidingDataset(root=['./datasets/ImageNet/ds',
                                      './datasets/ImageNet/HiNet',
                                      './datasets/ImageNet/TRM',
                                      './datasets/ImageNet/ISN',
                                      ],
                                type='val', trainFold_size=5, valFold_size=1, is_aug=False, auto_pad=auto_pad,
                                load_cover_secret=False)

    trainLoader = DataLoader(dataset=trainSet, batch_size=train_bz, num_workers=num_workers,
                             worker_init_fn=worker_init_fn, shuffle=True)
    valLoader = DataLoader(dataset=valSet, batch_size=val_bz, num_workers=num_workers, shuffle=False)

    train_revLosses = AverageMeter()
    val_revLosses = AverageMeter()
    train_logger = Logger(model_name=log_file, header=['epoch', 'RevLoss', 'lr'])

    for epoch in range(cur_epoch, max_epoch):
        # ---- train process ---- #
        kef.train()
        train_process = tqdm(trainLoader, mininterval=0)
        cur_train_batch = 0
        train_revLosses.reset()
        for idx, (container, revealSecret) in enumerate(train_process):
            cur_train_batch += len(container)
            container, revealSecret = container.cuda(), revealSecret.cuda()     
            optimizer.zero_grad()
            outputs = kef(container)
            revLoss = crit(outputs, revealSecret) # weighted loss: (perceptual, MSE, TV)
            revLoss.backward()
            optimizer.step()

            train_revLosses.update(revLoss.item(), container.shape[0])

            train_process.set_description(
                'Train, Epoch: {} | Batch: {}/{} | L2 Loss: {}'.format(
                    epoch + 1, cur_train_batch, trainSet.__len__(), train_revLosses.avg)
            )
        scheduler.step(train_revLosses.avg)

        # save train log
        train_logger.log(phase="train", values={
            'epoch': epoch + 1,
            'RevLoss': format(train_revLosses.avg, '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
        # save last weight
        save_states_path = os.path.join(save_dir, 'last_epoch.pth')
        states = {
            'epoch': epoch + 1,
            'state_dict': kef.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_states_path)

        # ---- val process ---- #
        kef.eval()
        val_process = tqdm(valLoader, mininterval=0)
        cur_val_batch = 0
        val_revLosses.reset()
        with torch.no_grad():
            for idx, (container, revealSecret) in enumerate(val_process):
                cur_val_batch += len(container)
                container, revealSecret = container.cuda(), revealSecret.cuda()
                outputs = kef(container)
                revLoss = crit(outputs, revealSecret) # 采用perceptual, MSE, TV的混合loss

                val_revLosses.update(revLoss.item(), container.shape[0])

                val_process.set_description(
                    'Val, Epoch: {} | Batch: {}/{} | L2 Loss: {}'.format(
                        epoch + 1, cur_val_batch, valSet.__len__(), val_revLosses.avg)
                )

            # save val log
            train_logger.log(phase="val", values={
                'epoch': epoch + 1,
                'RevLoss': format(val_revLosses.avg, '.4f'),
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # save better weight
            if epoch == 0 or val_revLosses.avg <= min_val_loss:
                min_val_loss = val_revLosses.avg
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_states_path = os.path.join(save_dir, 'epoch_{0}_{1:.4f}.pth'.format(epoch+1, val_revLosses.avg))
                states = {
                    'epoch': epoch + 1,
                    'state_dict': kef.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_states_path)




