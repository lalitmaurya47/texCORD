"""
train_texCORD.py


"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import utils.visualization as viz
from utils.logging import Logger
from utils import set_random_seed
from models.parts import TrainingOutput
from utils.metrics import texture_losses
from datasets.texture_data import build_dataloaders
from utils.scheduler import CosineAnnealingLRWithWarmup

from model.usfm import VisionTransformer


# -----------------------------------------------------------------------------
# Student model imports
# -----------------------------------------------------------------------------

try:
    from model.texCORD import (
       
        PVTv2B0TextureClusteringModel,
        ResNetUNetTextureClusteringModel,
    )
except Exception:
  
    PVTv2B0TextureClusteringModel = None
    ResNetUNetTextureClusteringModel = None


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

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


@dataclass
class TeacherPrediction:
    feat1_aug1: torch.Tensor
    feat1_aug2: torch.Tensor
    feat2_aug1: torch.Tensor
    feat2_aug2: torch.Tensor


# -----------------------------------------------------------------------------
# Foreground mask helper
# -----------------------------------------------------------------------------

def make_foreground_mask(
    x: torch.Tensor,
    threshold: float = 1e-3,
    normalized_black_value: Optional[float] = -1.0,
) -> torch.Tensor:
    """
    Create ultrasound foreground mask.

    If images are normalized to [-1, 1] and black background is -1,
    use normalized_black_value=-1.0.

    Returns:
        fg: [B, 1, H, W], float tensor
    """

    if normalized_black_value is not None:
        fg = (torch.abs(x - normalized_black_value) > threshold).float()
    else:
        fg = (torch.abs(x) > threshold).float()

    return fg


# -----------------------------------------------------------------------------
# USFM teacher loading
# -----------------------------------------------------------------------------

class USFMTeacherWrapper(nn.Module):
    """
    Wraps USFM ViT and returns patch-token feature map [B, C, H, W].
    """

    def __init__(self, usfm: nn.Module):
        super().__init__()
        self.usfm = usfm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        if x.shape[-2:] != (224, 224):
            x = F.interpolate(
                x,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

        x = self.usfm.patch_embed(x)

        if x.ndim != 3:
            raise RuntimeError(
                f"Expected USFM patch_embed output [B, N, C], got {x.shape}"
            )

        B, N, C = x.shape

        cls_tokens = self.usfm.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if getattr(self.usfm, "pos_embed", None) is not None:
            x = x + self.usfm.pos_embed

        if hasattr(self.usfm, "pos_drop"):
            x = self.usfm.pos_drop(x)

        for blk in self.usfm.blocks:
            x = blk(x)

        if hasattr(self.usfm, "norm") and self.usfm.norm is not None:
            x = self.usfm.norm(x)

        patch_tokens = x[:, 1:, :]

        side = int(patch_tokens.shape[1] ** 0.5)
        if side * side != patch_tokens.shape[1]:
            raise RuntimeError(
                f"Cannot reshape {patch_tokens.shape[1]} tokens into square map"
            )

        feat_map = patch_tokens.transpose(1, 2).reshape(B, C, side, side)
        return feat_map


def build_usfm_teacher(train_args, device: torch.device) -> Tuple[nn.Module, int]:
    """
    Build frozen USFM teacher.

    Expected train_args fields:
        usfm_checkpoint
        usfm_embed_dim
    """

    checkpoint_path = getattr(
        train_args,
        "usfm_checkpoint",
        "/users/lalit47/usfm/tusa/experiments/us-texture/teacher/USFM_latest.pth",
    )
    teacher_dim = int(getattr(train_args, "usfm_embed_dim", 768))

    usfm = VisionTransformer(norm_layer=partial(nn.LayerNorm, eps=1e-6))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    load_info = usfm.load_state_dict(checkpoint, strict=False)

    print("USFM teacher loaded.")
    print("Missing keys:", load_info.missing_keys)
    print("Unexpected keys:", load_info.unexpected_keys)

    if len(load_info.missing_keys) > 0 or len(load_info.unexpected_keys) > 0:
        print(
            "\nWARNING: USFM checkpoint was loaded with missing/unexpected keys. "
            "For reliable KD, make sure the checkpoint architecture exactly matches "
            "the instantiated VisionTransformer.\n"
        )

    teacher = USFMTeacherWrapper(usfm).to(device)
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    return teacher, teacher_dim


# -----------------------------------------------------------------------------
# Student builder
# -----------------------------------------------------------------------------

def build_student_model(config, device: torch.device) -> nn.Module:
    """
    Build student model.

    Expected config.TextureClusteringModel.student_backbone:
        - "pvt"
        - "resnet_unet"
    """

    model_cfg = config.TextureClusteringModel
    backbone = getattr(model_cfg, "student_backbone", "pvt")

    if backbone == "pvt":
        if PVTv2B0TextureClusteringModel is None:
            raise ImportError("Could not import PVTv2B0TextureClusteringModel.")
        return PVTv2B0TextureClusteringModel(model_cfg).to(device)

    if backbone == "resnet_unet":
        if ResNetUNetTextureClusteringModel is None:
            raise ImportError("Could not import ResNetUNetTextureClusteringModel.")
        return ResNetUNetTextureClusteringModel(model_cfg).to(device)

    raise ValueError(f"Unknown student_backbone: {backbone}")


# -----------------------------------------------------------------------------
# Distillation losses
# -----------------------------------------------------------------------------

class StudentTeacherProjection(nn.Module):
    """
    1x1 projection to align student feature channels to teacher feature channels.
    LazyConv2d initializes automatically at first forward pass.
    """

    def __init__(self, teacher_dim: int):
        super().__init__()
        self.proj = nn.LazyConv2d(teacher_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GlobalKDLoss(nn.Module):
    """
    Global feature distillation by average pooling.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        foreground_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        teacher_features = F.interpolate(
            teacher_features,
            size=student_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if foreground_mask is not None:
            fg = F.interpolate(
                foreground_mask,
                size=student_features.shape[-2:],
                mode="nearest",
            )

            fs = (student_features * fg).sum(dim=(-2, -1))
            ft = (teacher_features * fg).sum(dim=(-2, -1))
            denom = fg.sum(dim=(-2, -1)).clamp_min(1.0)

            fs = fs / denom
            ft = ft / denom
        else:
            fs = F.adaptive_avg_pool2d(student_features, 1).flatten(1)
            ft = F.adaptive_avg_pool2d(teacher_features, 1).flatten(1)

        if self.normalize:
            fs = F.normalize(fs, dim=1)
            ft = F.normalize(ft, dim=1)

        return F.mse_loss(fs, ft)


class TextureGuidedKDLoss(nn.Module):
    """
    Foreground-aware texture-guided regional distillation.

    Inputs:
        student_features: [B, C, Hs, Ws]
        teacher_features: [B, C, Ht, Wt]
        texture_probs:    [B, K, H, W]
        foreground_mask:  [B, 1, H, W], optional
    """

    def __init__(self, normalize: bool = True, eps: float = 1e-6):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        texture_probs: torch.Tensor,
        foreground_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_size = student_features.shape[-2:]

        teacher_features = F.interpolate(
            teacher_features,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        texture_probs = F.interpolate(
            texture_probs,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        if foreground_mask is not None:
            foreground_mask = F.interpolate(
                foreground_mask,
                size=target_size,
                mode="nearest",
            )
            texture_probs = texture_probs * foreground_mask

        if student_features.shape[1] != teacher_features.shape[1]:
            raise ValueError(
                f"Feature channel mismatch: student={student_features.shape[1]}, "
                f"teacher={teacher_features.shape[1]}. Use projection first."
            )

        weights = texture_probs / (
            texture_probs.sum(dim=(-2, -1), keepdim=True) + self.eps
        )

        losses = []

        for k in range(weights.shape[1]):
            wk = weights[:, k:k + 1]

            valid = wk.sum(dim=(-2, -1)).squeeze(1) > self.eps
            if valid.sum() == 0:
                continue

            fs_k = (student_features * wk).sum(dim=(-2, -1))
            ft_k = (teacher_features * wk).sum(dim=(-2, -1))

            fs_k = fs_k[valid]
            ft_k = ft_k[valid]

            if self.normalize:
                fs_k = F.normalize(fs_k, dim=1)
                ft_k = F.normalize(ft_k, dim=1)

            losses.append(F.mse_loss(fs_k, ft_k))

        if len(losses) == 0:
            return student_features.sum() * 0.0

        return torch.stack(losses).mean()


class DistillationLossBundle(nn.Module):
    """
    Computes global KD and texture-guided KD over all four TUSA augmented views.
    """

    def __init__(
        self,
        projection: nn.Module,
        lambda_global_kd: float = 0.5,
        lambda_texture_kd: float = 1.0,
        detach_texture_for_kd: bool = True,
        use_foreground_mask: bool = True,
        foreground_global_kd: bool = True,
        normalized_black_value: Optional[float] = -1.0,
    ):
        super().__init__()
        self.projection = projection
        self.lambda_global_kd = lambda_global_kd
        self.lambda_texture_kd = lambda_texture_kd
        self.detach_texture_for_kd = detach_texture_for_kd
        self.use_foreground_mask = use_foreground_mask
        self.foreground_global_kd = foreground_global_kd
        self.normalized_black_value = normalized_black_value

        self.global_kd = GlobalKDLoss()
        self.texture_kd = TextureGuidedKDLoss()

    def _single_view_loss(
        self,
        pred: TrainingOutput,
        teacher_feat: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pred.features is None:
            raise RuntimeError(
                "Student output has features=None. Call model(x, return_features=True), "
                "or modify the student model to return feature maps."
            )

        fs = self.projection(pred.features)
        ft = teacher_feat.detach()

        foreground_mask = None
        if self.use_foreground_mask or self.foreground_global_kd:
            foreground_mask = make_foreground_mask(
                image,
                threshold=1e-3,
                normalized_black_value=self.normalized_black_value,
            )

        global_fg = foreground_mask if self.foreground_global_kd else None
        texture_fg = foreground_mask if self.use_foreground_mask else None

        loss_global = self.global_kd(
            fs,
            ft,
            foreground_mask=global_fg,
        )

        texture_probs = pred.structure
        if self.detach_texture_for_kd:
            texture_probs = texture_probs.detach()

        loss_texture = self.texture_kd(
            fs,
            ft,
            texture_probs,
            foreground_mask=texture_fg,
        )

        return loss_global, loss_texture

    def forward(
        self,
        pred: SSLPrediction,
        teacher_pred: TeacherPrediction,
        images: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:

        pairs = [
            (pred.pred1_aug1, teacher_pred.feat1_aug1, images[0]),
            (pred.pred1_aug2, teacher_pred.feat1_aug2, images[1]),
            (pred.pred2_aug1, teacher_pred.feat2_aug1, images[2]),
            (pred.pred2_aug2, teacher_pred.feat2_aug2, images[3]),
        ]

        global_losses = []
        texture_losses_ = []

        for p, t, img in pairs:
            lg, lt = self._single_view_loss(p, t, img)
            global_losses.append(lg)
            texture_losses_.append(lt)

        loss_global = torch.stack(global_losses).mean()
        loss_texture = torch.stack(texture_losses_).mean()

        total = (
            self.lambda_global_kd * loss_global
            + self.lambda_texture_kd * loss_texture
        )

        metrics = {
            "global_kd": float(loss_global.detach().cpu()),
            "texture_kd": float(loss_texture.detach().cpu()),
            "kd_total": float(total.detach().cpu()),
        }

        return total, metrics


# -----------------------------------------------------------------------------
# Forward helper
# -----------------------------------------------------------------------------

def forward_student(model: nn.Module, x: torch.Tensor) -> TrainingOutput:
    """
    Supports:
        model(x, return_features=True)
    and older:
        model(x)
    """

    try:
        return model(x, return_features=True)
    except TypeError:
        out = model(x)
        if not hasattr(out, "features") or out.features is None:
            raise RuntimeError(
                "Your student model must return feature maps for KD. "
                "Modify the model forward to include features in TrainingOutput."
            )
        return out


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train_loop(
    student: nn.Module,
    teacher: nn.Module,
    train_dl,
    ssl_losses,
    kd_losses: DistillationLossBundle,
    optimizer,
    scaler,
    device,
):
    student.train()
    teacher.eval()
    kd_losses.train()

    kd_metric_accumulator = {
        "global_kd": 0.0,
        "texture_kd": 0.0,
        "kd_total": 0.0,
    }

    loss_metric_accumulator = {
        "loss_ssl": 0.0,
        "loss_kd": 0.0,
        "loss_total": 0.0,
    }

    for batch in train_dl:
        x1_aug1 = batch["x1_aug1"].to(device, non_blocking=True)
        x1_aug2 = batch["x1_aug2"].to(device, non_blocking=True)
        x2_aug1 = batch["x2_aug1"].to(device, non_blocking=True)
        x2_aug2 = batch["x2_aug2"].to(device, non_blocking=True)

        x1 = batch["x1"].to(device, non_blocking=True)
        x2 = batch["x2"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            t1_aug1 = teacher(x1_aug1)
            t1_aug2 = teacher(x1_aug2)
            t2_aug1 = teacher(x2_aug1)
            t2_aug2 = teacher(x2_aug2)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            y1_aug1 = forward_student(student, x1_aug1)
            y1_aug2 = forward_student(student, x1_aug2)
            y2_aug1 = forward_student(student, x2_aug1)
            y2_aug2 = forward_student(student, x2_aug2)

            ssl_pred = SSLPrediction(
                y1_aug1,
                y1_aug2,
                y2_aug1,
                y2_aug2,
            )

            ssl_label = SSLLabel(x1, x2)

            teacher_pred = TeacherPrediction(
                t1_aug1,
                t1_aug2,
                t2_aug1,
                t2_aug2,
            )

            loss_ssl = ssl_losses(ssl_pred, ssl_label)

            loss_kd, kd_metrics = kd_losses(
                ssl_pred,
                teacher_pred,
                images=(x1_aug1, x1_aug2, x2_aug1, x2_aug2),
            )

            loss = loss_ssl + loss_kd

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_metrics = {
            "loss_ssl": float(loss_ssl.detach().cpu()),
            "loss_kd": float(loss_kd.detach().cpu()),
            "loss_total": float(loss.detach().cpu()),
        }

        for key in kd_metric_accumulator:
            kd_metric_accumulator[key] += kd_metrics[key]

        for key in loss_metric_accumulator:
            loss_metric_accumulator[key] += loss_metrics[key]

    metrics = ssl_losses.finish_epoch(len(train_dl))

    for key, value in kd_metric_accumulator.items():
        metrics[key] = value / max(len(train_dl), 1)

    for key, value in loss_metric_accumulator.items():
        metrics[key] = value / max(len(train_dl), 1)

    metrics["lr"] = optimizer.param_groups[0]["lr"]

    return metrics


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

@torch.no_grad()
def val(
    model,
    val_ds,
    colors,
    device,
    n: int = 16,
    normalized_black_value: Optional[float] = -1.0,
):
    model.eval()

    ims = val_ds.stacked[:n].to(device)
    pred = forward_student(model, ims)

    fg = make_foreground_mask(
        ims,
        threshold=1e-3,
        normalized_black_value=normalized_black_value,
    )

    val_recon_l1 = (
        torch.abs(pred.reconstruction - ims) * fg
    ).sum() / fg.sum().clamp_min(1.0)

    val_recon_l1 = val_recon_l1.item()

    eps = 1e-8
    p = pred.structure.clamp_min(eps)

    fg_for_p = F.interpolate(
        fg,
        size=p.shape[-2:],
        mode="nearest",
    )

    fg_sum = fg_for_p.sum().clamp_min(1.0)

    channel_usage = (p * fg_for_p).sum(dim=(0, 2, 3)) / fg_sum

    val_global_entropy = -(
        channel_usage * torch.log(channel_usage + eps)
    ).sum().item()

    pixel_entropy_map = -(p * torch.log(p + eps)).sum(dim=1, keepdim=True)

    val_pixel_entropy = (
        pixel_entropy_map * fg_for_p
    ).sum() / fg_sum

    val_pixel_entropy = val_pixel_entropy.item()

    val_usage_min = channel_usage.min().item()
    val_usage_max = channel_usage.max().item()
    val_usage_std = channel_usage.std().item()

    hard_mask = pred.structure.argmax(dim=1)

    val_im = viz.texture_val_im(
        ims,
        hard_mask,
        pred.reconstruction,
        colors,
    )

    return val_im, {
        "val_recon_l1_fg": val_recon_l1,
        "val_global_entropy_fg": val_global_entropy,
        "val_pixel_entropy_fg": val_pixel_entropy,
        "val_usage_min_fg": val_usage_min,
        "val_usage_max_fg": val_usage_max,
        "val_usage_std_fg": val_usage_std,
    }


# -----------------------------------------------------------------------------
# Main training entry
# -----------------------------------------------------------------------------

def run(train_args, config):
    logger = Logger(train_args, config)
    set_random_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, val_ds = build_dataloaders(train_args, config.dataset)

    student = build_student_model(config, device)
    teacher, teacher_dim = build_usfm_teacher(train_args, device)

    projection = StudentTeacherProjection(teacher_dim).to(device)

    colors = viz.color_dict(student.dict_size, device)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projection.parameters()),
        lr=train_args.lr,
        weight_decay=getattr(train_args, "weight_decay", 1e-4),
    )

    scheduler = CosineAnnealingLRWithWarmup(
        optimizer,
        warmup=getattr(train_args, "warmup", 10),
        eta_min=train_args.lr * getattr(train_args, "eta_min_ratio", 1e-2),
        T_max=train_args.epochs,
    )

    scaler = torch.amp.GradScaler(
        "cuda" if device.type == "cuda" else "cpu"
    )

    ssl_losses = texture_losses(scheduler, device)

    normalized_black_value = getattr(train_args, "normalized_black_value", -1.0)
    if normalized_black_value is not None:
        normalized_black_value = float(normalized_black_value)

    kd_losses = DistillationLossBundle(
        projection=projection,
        lambda_global_kd=float(getattr(train_args, "lambda_global_kd", 0.5)),
        lambda_texture_kd=float(getattr(train_args, "lambda_texture_kd", 1.0)),
        detach_texture_for_kd=bool(getattr(train_args, "detach_texture_for_kd", True)),
        use_foreground_mask=bool(getattr(train_args, "use_foreground_mask", True)),
        foreground_global_kd=bool(getattr(train_args, "foreground_global_kd", True)),
        normalized_black_value=normalized_black_value,
    ).to(device)

    start_epoch = 1

    if logger.continued:
        checkpoint = torch.load(logger.model_path, map_location=device, weights_only=False)     ### did moficitaion  weights_only=False

        student.load_state_dict(checkpoint["student_state"])
        projection.load_state_dict(checkpoint["projection_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed training from epoch {start_epoch}")

    checkpoint_dir = os.path.dirname(logger.model_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for e in range(start_epoch, train_args.epochs + 1, 5):
        with tqdm(range(5), desc=f"Epoch #{e}/{train_args.epochs}") as progress_bar:
            for k in progress_bar:
                current_epoch = e + k

                if current_epoch > train_args.epochs:
                    break

                metrics = train_loop(
                    student=student,
                    teacher=teacher,
                    train_dl=train_dl,
                    ssl_losses=ssl_losses,
                    kd_losses=kd_losses,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                )

                val_im, val_metrics = val(
                    student,
                    val_ds,
                    colors,
                    device,
                    normalized_black_value=normalized_black_value,
                )

                metrics.update(val_metrics)

                logger.log(current_epoch, metrics, val_im)

                progress_bar.set_postfix(metrics)

                scheduler.step()

                if current_epoch % 5 == 0 or current_epoch == train_args.epochs:
                    checkpoint = {
                        "epoch": current_epoch,
                        "student_state": student.state_dict(),
                        "projection_state": projection.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "config": config,
                    }
                
                    epoch_path = os.path.join(
                        checkpoint_dir,
                        f"checkpoint_epoch_{current_epoch:04d}.pth",
                    )
                
                    torch.save(checkpoint, epoch_path)
                    torch.save(checkpoint, logger.model_path)

    final_checkpoint = {
        "student_state": student.state_dict(),
        "projection_state": projection.state_dict(),
        "config": config,
    }

    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(final_checkpoint, final_path)

    print(f"Final model saved to: {final_path}")