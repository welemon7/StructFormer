# coding=gb2312
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage, RandomRotation, RandomCrop, ColorJitter
from PIL import Image
import warnings
from torchvision import transforms
import torch
import random
import os
import numpy as np
import cv2
warnings.filterwarnings("ignore")


class SRDTrainDataset(Dataset):
    """ GaussianBlur mask """
    def __init__(self, data_dir, img_size=256, augment=True):
        self.A_dir = os.path.join(data_dir, 'shadow')
        self.B_dir = os.path.join(data_dir, 'non_shadow')
        self.C_dir = os.path.join(data_dir, 'mask')
        self.image_files = [f for f in os.listdir(self.A_dir) if f.endswith(('.jpg', '.png'))]
        self.img_size = img_size
        self.augment = augment

        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                RandomRotation(degrees=15),
                RandomCrop(size=img_size),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])
        else:
            self.augment_transform = None

        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        A_path = os.path.join(self.A_dir, img_name)
        B_path = os.path.join(self.B_dir, img_name)
        C_path = os.path.join(self.C_dir, img_name)

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('L')

        A = self.base_transform(A_img)
        B = self.base_transform(B_img)
        C = self.base_transform(C_img)


        if self.augment_transform:
            seed = random.randint(0, 2 ** 32)

            random.seed(seed)
            torch.manual_seed(seed)
            A = self.augment_transform(A)

            random.seed(seed)
            torch.manual_seed(seed)
            B = self.augment_transform(B)

            random.seed(seed)
            torch.manual_seed(seed)
            C = self.augment_transform(C)

        return A, B, C


# ---------------------------------------------------------------------------------------------------------------------
class SRDTestDataset(Dataset):
    def __init__(self, data_dir, img_size=256, augment=True):
        self.A_dir = os.path.join(data_dir, 'shadow')
        self.B_dir = os.path.join(data_dir, 'non_shadow')
        self.C_dir = os.path.join(data_dir, 'mask')
        self.image_files = [f for f in os.listdir(self.A_dir) if f.endswith(('.jpg', '.png'))]
        self.img_size = img_size
        self.augment = augment

        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        A_path = os.path.join(self.A_dir, img_name)
        B_path = os.path.join(self.B_dir, img_name)
        C_path = os.path.join(self.C_dir, img_name)

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('L')

        A = self.base_transform(A_img)
        B = self.base_transform(B_img)
        C = self.base_transform(C_img)

        return A, B, C
