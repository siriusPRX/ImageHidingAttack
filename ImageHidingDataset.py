import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import os

def reflection_padding(img, patch_size):
    h, w = img.shape[1:3]
    h_pad = (patch_size[0] - h % patch_size[0]) % patch_size[0]
    w_pad = (patch_size[1] - w % patch_size[1]) % patch_size[1]

    top_pad = h_pad // 2
    down_pad = h_pad - top_pad
    left_pad = w_pad // 2
    right_pad = w_pad - left_pad
    return np.pad(img, ((0, 0), (top_pad, down_pad), (left_pad, right_pad), (0, 0)), 'reflect')
    
def divide_patch_overlap(img, patch_size, patch_factor=1):
    stride = (patch_size[0]//patch_factor, patch_size[1]//patch_factor)
    imgs = []
    h, w = img.shape[1:3]
    for i in range(patch_size[0], h+1, stride[0]):
        for j in range(patch_size[1], w+1, stride[1]):
            imgs.append(img[:, i - patch_size[0]:i, j - patch_size[1]:j, :])
    imgs = np.vstack(imgs)
    return imgs

def pad_and_divide(img, patch_size, patch_factor=1):
    img = img[np.newaxis, ::]
    img = reflection_padding(img, patch_size)
    patches = divide_patch_overlap(img, patch_size, patch_factor)
    return patches


class ImageHidingDataset(Dataset):
    def __init__(self, root, type='train', trainFold_size=200, valFold_size=100, testFold_size=1000,
                 is_aug=False, auto_pad=True, patch_size=(64, 64), patch_factor=1, load_cover_secret=False):
        super(ImageHidingDataset, self).__init__()
        self.containers_path = []
        self.is_aug = is_aug
        self.auto_pad = auto_pad
        self.patch_size = patch_size
        self.patch_factor = patch_factor
        self.load_cover_secret = load_cover_secret

        # Partitioning the dataset
        np.random.seed(2022)
        for idx, fold in enumerate(root):
            path = sorted(glob.glob(os.path.join(fold, 'containers/*.png')))
            np.random.shuffle(path)
            if type != 'test':
                assert (trainFold_size + valFold_size) <= len(path), 'the sub-fold size over whole fold size'
            if type == 'train':
                self.containers_path += path[:trainFold_size]
            elif type == 'val':
                self.containers_path += path[trainFold_size:trainFold_size+valFold_size]
            elif type == 'test':
                self.containers_path += path[-testFold_size:]
        np.random.seed(None)
        print('{}, Total Images:{}'.format(type, len(self.containers_path)))

        # The images of different sizes in the training set were padded and divided
        if self.auto_pad:
            print('-----Data AutoPad-----')
            if self.load_cover_secret:
                self.covers = []
                self.secrets = []
                for container_path in self.containers_path:
                    cover_path = container_path.replace('container', 'cover')
                    secret_path = container_path.replace('container', 'secret')
                    cover = cv2.cvtColor(cv2.imread(cover_path), cv2.COLOR_BGR2RGB)
                    secret = cv2.cvtColor(cv2.imread(secret_path), cv2.COLOR_BGR2RGB)
                    cover = pad_and_divide(cover, self.patch_size, patch_factor=self.patch_factor)
                    secret = pad_and_divide(secret, self.patch_size, patch_factor=self.patch_factor)
                    self.covers.append(cover)
                    self.secrets.append(secret)
                self.covers = np.vstack(self.covers)
                self.secrets = np.vstack(self.containers)

            self.containers = []
            self.revealedSecrets = []
            for container_path in self.containers_path:
                revealedSecret_path = container_path.replace('container', 'revealedSecret')
                container = cv2.cvtColor(cv2.imread(container_path), cv2.COLOR_BGR2RGB)
                revealedSecret = cv2.cvtColor(cv2.imread(revealedSecret_path), cv2.COLOR_BGR2RGB)
                container = pad_and_divide(container, self.patch_size, patch_factor=self.patch_factor)
                revealedSecret = pad_and_divide(revealedSecret, self.patch_size, patch_factor=self.patch_factor)
                self.containers.append(container)
                self.revealedSecrets.append(revealedSecret)
            self.containers = np.vstack(self.containers)
            self.revealedSecrets = np.vstack(self.revealedSecrets)

        # transform
        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def __getitem__(self, idx):
        if self.auto_pad:
            if self.load_cover_secret:
                cover, secret = self.covers[idx], self.secrets[idx]
            container, revealedSecret = self.containers[idx], self.revealedSecrets[idx]
        else:
            if self.load_cover_secret:
                cover_path = self.containers_path[idx].replace('container', 'cover')
                secret_path = self.containers_path[idx].replace('container', 'secret')
                cover = cv2.cvtColor(cv2.imread(cover_path), cv2.COLOR_BGR2RGB)
                secret = cv2.cvtColor(cv2.imread(secret_path), cv2.COLOR_BGR2RGB)
                cover = cover.astype('float32') / 255.
                secret = secret.astype('float32') / 255.
            revealedSecret_path = self.containers_path[idx].replace('container', 'revealedSecret')
            container = cv2.cvtColor(cv2.imread(self.containers_path[idx]), cv2.COLOR_BGR2RGB)
            revealedSecret = cv2.cvtColor(cv2.imread(revealedSecret_path), cv2.COLOR_BGR2RGB)
            container = container.astype('float32') / 255.
            revealedSecret = revealedSecret.astype('float32') / 255.

        '''
        if self.is_aug:
            container, revealedSecret = self.random_flip_and_rot90(container, revealedSecret, p=0.2)
            
            choice = np.random.choice(2, 1)
            if choice == 0:
                container = self.gaussion_noise(container, mean=0, var=0.001, p=0.2)
            elif choice == 1:
                container = self.resample(container, times=2, p=0.5)
            if choice == 2:
                container, revealedSecret = self.random_flip_and_rot90(container, revealedSecret, p=0.2)
        '''
        if self.load_cover_secret:
            return self.transform(cover), self.transform(secret), \
                   self.transform(container), self.transform(revealedSecret)
        return self.transform(container), self.transform(revealedSecret)

    # Random Flip and Rotate
    def random_flip_and_rot90(self, container, revealSecret, is_vflip=True, is_hflip=True, is_rot=True, p=0.5):
        # axis=0 Vertical Flip, axis=1 Horizontal Flip
        if is_vflip and np.random.random() < p:
            container = np.flip(container, axis=0)
            revealSecret = np.flip(revealSecret, axis=0)
        if is_hflip and np.random.random() < p:
            container = np.flip(container, axis=1)
            revealSecret = np.flip(revealSecret, axis=1)
        if is_rot and np.random.random() < p:
            rot_num = np.random.randint(1, 4)
            container = np.rot90(container, k=rot_num, axes=(0, 1))
            revealSecret = np.rot90(revealSecret, k=rot_num, axes=(0, 1))

        return container, revealSecret

    def gaussion_noise(self, image, mean=0, var=0.001, p=0.5):
        if np.random.random() < p:
            noise = np.random.normal(mean, var ** 0.5, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1.0)
        return image

    def resample(self, image, times=2, p=0.5):
        if np.random.random() < p:
            h, w = image.shape[:2]
            image = cv2.resize(image, (w // times, h // times))
            image = cv2.resize(image, (w, h))
        return image

    def __len__(self):
        if self.auto_pad:
            return self.containers.shape[0]
        return len(self.containers_path)


if __name__ == '__main__':
    trainSet = ImageHidingDataset(root=['/data1/pengrx/DeepSteg/deepsteg_v1/data/224x/train/',
                                        '/data1/pengrx/DeepSteg/deepsteg_v2/data/224x/train/'], type='train',
                                  trainFold_size=10000, valFold_size=200)
    valSet = ImageHidingDataset(root=['/data1/pengrx/DeepSteg/deepsteg_v1/data/224x/train/',
                                      '/data1/pengrx/DeepSteg/deepsteg_v2/data/224x/train/'], type='val',
                                trainFold_size=10000, valFold_size=200)

    trainLoader = DataLoader(dataset=trainSet, batch_size=32, shuffle=True)
    valLoader = DataLoader(dataset=valSet, batch_size=1, shuffle=False)

    for container, revealSecret in trainLoader:
        print()



