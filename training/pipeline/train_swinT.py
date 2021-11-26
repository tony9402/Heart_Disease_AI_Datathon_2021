import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import transforms

import sys
sys.path.append('/github/training')

from model.UNet import UNet
from model.swinT import SwinTransformer
from utils.HeartDiseaseDataLoader import HeartDiseaseDataset
from utils.dice_score import dice_loss
from utils.evaluate import evaluate


if __name__ == '__main__':

    train_transforms = A.Compose([
        A.Resize(width=224, height=224, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.25),
        A.IAAEmboss(p=0.25),
        A.Blur(p=0.01, blur_limit=3),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8),
        A.Normalize(p=1.0),
        transforms.ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.Resize(width=224, height=224, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        transforms.ToTensorV2()
    ])

    train_dataset = HeartDiseaseDataset('./data/train', train_transforms)
    val_dataset = HeartDiseaseDataset('./data/validation', val_transforms)

    batch_size = 1
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=1, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = UNet(n_channels=3, n_classes=2, bilinear=True) 
    model = SwinTransformer(num_classes=2)
    model.to(device=device)

    learning_rate = 0.0001
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    epochs = 50
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, total=len(train_loader))
        for images, masks in pbar:
            #images = batch['image']
            #masks = batch['mask']

            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=True):
                masks_pred = model(images)
                loss = criterion(masks_pred, masks) \
                    + dice_loss(F.softmax(masks_pred, dim=1).float(),
                              F.one_hot(masks, model.n_classes).permute(0, 3, 1, 2).float(),
                              multiclass=False)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()

            #division_step = (epoch// (10 * batch_size))
        if True:
            #if global_step % division_step == 0:
            histograms = {}
            for tag, value in model.named_parameters():
                tag = tag.replace('/', '.')

            val_score = evaluate(model, val_loader, device)
            scheduler.step(val_score)
            print(val_score.cpu())
