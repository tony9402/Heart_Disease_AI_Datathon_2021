import os
import sys
sys.path.append('/github/')

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import transforms

import wandb

from training.config import load_config
from training.model import get_model
from training.tools import optimizer as op
from training.tools import losses
from training.utils.HeartDiseaseDataLoader import HeartDiseaseDataset

def create_train_transforms(size=128):
    return A.Compose([
        A.Resize(width=size, height=size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.25),
        A.Blur(p=0.5),
        A.ElasticTransform(alpha=0.3, p=0.2),
        A.Rotate(15, p=0.5),
        transforms.ToTensorV2()
    ])

def create_val_transforms(size=128):
    return A.Compose([
        A.Resize(width=size, height=size, p=1.0),
        A.HorizontalFlip(p=0.5),
        transforms.ToTensorV2()
    ])

def load_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--config', default="UNet")
    arg('--data', default="/github/data.csv")
    arg('--model_save', default='/github/model_result/UNet/')
    arg('--model_save_prefix', default="UNet_")
    return parser.parse_args()

def main():
    args = load_args()
    config = load_config(args.config)
    model_f = get_model(config.model)

    train_dataset = HeartDiseaseDataset(args.data, create_train_transforms(config.size), True)
    val_dataset = HeartDiseaseDataset(args.data, create_val_transforms(config.size), False)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_f(**config.model_config).to(device)
    wandb.watch(model)

    optimizer, scheduler = op.create_optimizer(config.optimizer, model)
    loss_f = losses.load_loss(config.loss)

    global_step = 0
    epochs = config.optimizer.scheduler.epochs
    best_score = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        current_loss = 0
        for idx, (images, masks) in pbar:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(images).squeeze(1)
            pred = torch.sigmoid(pred)
            loss = loss_f(pred, masks)
            current_loss += loss.item()
            pbar.set_postfix({
                'loss': current_loss / (idx + 1),
                'lr': f"{scheduler.get_last_lr()[0]:.5f}"
            })

            if (idx + 1) % 50 == 0:
                wandb.log({
                    'loss': current_loss / (idx + 1),
                    'lr': float(scheduler.get_last_lr()[0])
                })

            loss.backward()
            optimizer.step()
            scheduler.step()

        pbar.close()

        # Validation
        model.eval()
        score = 0

        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for idx, (images, masks) in pbar:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            with torch.no_grad():
                pred = model(images).squeeze(1)
                pred = torch.sigmoid(pred)
                pred = (pred > 0.5).float()
                cur_score = losses.dice_metric(pred, masks).item()
                score = (score * idx + cur_score) / (idx + 1)
                pbar.set_postfix({
                    "DICE": score,
                    "JAC": score / (2 - score)
                })
        pbar.close()
        wandb.log({
            "DICE": score,
            "JAC": score / (2 - score)
        })

        os.makedirs(args.model_save, exist_ok=True)
        if best_score < score:
            best_score = score
            path = os.path.join(args.model_save, f"{args.model_save_prefix}best.pth")
            torch.save(model.state_dict(), path)

        path = os.path.join(args.model_save, f"{args.model_save_prefix}_lastest.pth")
        torch.save(model.state_dict(), path)

if __name__ == '__main__':
    wandb.init(project="HDAD", entity="tony9402")
    main()
