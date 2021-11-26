from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from glob import glob

class HeartDiseaseDataset(Dataset):
    def __init__(self, path, transforms):
        files = glob(f"{path}/**/*.*", recursive=True)
        self.images = sorted([file for file in files if file.endswith('.png')])
        self.masks = sorted([file for file in files if file.endswith('.npy')])

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
    h = HeartDiseaseDataset('./../../data/train', '')
