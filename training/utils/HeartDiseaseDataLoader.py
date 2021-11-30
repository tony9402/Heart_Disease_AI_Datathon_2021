from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd

"""
For DataLoader
"""
class HeartDiseaseDataset(Dataset):
    def __init__(self, data, transforms, train = True):
        data = pd.read_csv(data, header=None)

        self.images = list()
        self.masks = list()

        if train:
            self.images = data[data[3] == 'train'].iloc[:, 0].tolist()
            self.masks = data[data[3] == 'train'].iloc[:, 1].tolist()
        else:
            self.images = data[data[3] != 'train'].iloc[:, 0].tolist()
            self.masks = data[data[3] != 'train'].iloc[:, 1].tolist()

        if len(self.images) != len(self.masks):
            raise "len(images) != len(masks)"

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        mask  = np.load(self.masks[idx])

        augmentation = self.transforms(image = image, mask = mask)
        image = augmentation['image']
        mask  = augmentation['mask']

        return image, mask

if __name__ == '__main__':
    # Test Module
    # h = HeartDiseaseDataset('./../../data/train', '')
    pass