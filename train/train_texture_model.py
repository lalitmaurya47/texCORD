import os
from dataclasses import dataclass

import torch
from tqdm import tqdm

import utils.visualization as viz
from utils.logging import Logger
from utils import set_random_seed
from models.parts import TrainingOutput
from utils.metrics import texture_losses
from datasets.texture_data import build_dataloaders
from models.full_model import TextureClusteringModel
from utils.scheduler import CosineAnnealingLRWithWarmup


def run(train_args, config):
    logger = Logger(train_args, config)
    set_random_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dl, val_ds = build_dataloaders(train_args, config.dataset)
    model = TextureClusteringModel(config.TextureClusteringModel).to(device)
    colors = viz.color_dict(model.dict_size, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.lr)
    scheduler = CosineAnnealingLRWithWarmup(optimizer, warmup=10, eta_min=train_args.lr * 1e-2, T_max=train_args.epochs)
    scaler = torch.amp.GradScaler()
    losses = texture_losses(scheduler, device)

    start_epoch = 1
    if logger.continued:
        checkpoint = torch.load(logger.model_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1

        print(f"  Resumed training from epoch {start_epoch}")

    for e in range(start_epoch, train_args.epochs + 1, 5):
        with tqdm(range(5), desc=f"Epoch #{e}/{train_args.epochs}") as progress_bar:
            for k in progress_bar:
                metrics = train_loop(model, train_dl, losses, optimizer, scaler, device)
                val_im = val(model, val_ds, colors, device)
                logger.log(e + k, metrics, val_im)

                progress_bar.set_postfix(metrics)
                scheduler.step()

        if (e - 1) % 100 == 0:
            checkpoint = {
                "epoch": e,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
            }
            torch.save(checkpoint, logger.model_path)
    torch.save(model.state_dict(), logger.model_path)


@dataclass
class SSLPrediction:
    pred1_aug1: TrainingOutput
    pred1_aug2: TrainingOutput
    pred2_aug1: TrainingOutput
    pred2_aug2: TrainingOutput


@dataclass
class SSLLabel:
    x1: torch.Tensor
    x2: torch.Tensor


def train_loop(model, train_dl, losses, optimizer, scaler, device):
    model.train()
    for batch in train_dl:
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            y1_aug1 = model(batch['x1_aug1'].to(device))
            y1_aug2 = model(batch['x1_aug2'].to(device))
            y2_aug1 = model(batch['x2_aug1'].to(device))
            y2_aug2 = model(batch['x2_aug2'].to(device))

            loss = losses(
                SSLPrediction(y1_aug1, y1_aug2, y2_aug1, y2_aug2),
                SSLLabel(
                    batch['x1'].to(device),
                    batch['x2'].to(device),
                )
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return losses.finish_epoch(len(train_dl))


@torch.no_grad()
def val(model, val_ds, colors, device):
    model.eval()
    ims = val_ds.stacked.to(device)
    pred = model(ims)
    return viz.texture_val_im(ims, pred.structure.argmax(dim=1), pred.reconstruction, colors)
