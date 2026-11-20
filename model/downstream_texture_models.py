"""
downstream_texture_models.py

Downstream model wrappers for your pretrained texture-guided USFM-distilled
student model.

Supports:
  1. Segmentation: pretrained texture student encoder + lightweight FPN-like decoder
  2. Classification: pretrained texture student encoder + linear/MLP classifier

Compatible pretrained checkpoints from train_texguided_usfm_distill.py:
    checkpoint = {
        "student_state": student.state_dict(),
        "projection_state": projection.state_dict(),
        ...
    }

or final checkpoint:
    checkpoint = {
        "student_state": student.state_dict(),
        "projection_state": projection.state_dict(),
    }

Expected student backbones:
    - DeiTTextureClusteringModel
    - MobileNetTextureClusteringModel

Place this file where your downstream scripts can import it.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.tgd import DeiTTextureClusteringModel, MobileNetTextureClusteringModel
except Exception:
    # Alternative path if placed under models/
    from models.light_tusa_students import DeiTTextureClusteringModel, MobileNetTextureClusteringModel

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config helper
# -----------------------------------------------------------------------------

class AttrDict(dict):
    def __getattr__(self, item):
        value = self[item]
        if isinstance(value, dict):
            return AttrDict(value)
        return value


def make_student_config(args):
    """
    Build a small config object expected by DeiTTextureClusteringModel or
    MobileNetTextureClusteringModel.
    """
    return AttrDict({
        "dict_size": getattr(args, "dict_size", 5),
        "img_size": getattr(args, "img_size", 224),
        "latent_size": getattr(args, "latent_size", 256),
        "student_backbone": getattr(args, "student_backbone", "deit"),
        "deit": {
            "backbone": getattr(args, "deit_backbone", "deit_tiny_patch16_224"),
            "pretrained": False,
            "decoder_hidden": getattr(args, "decoder_hidden", 128),
        },
        "mobilenet": {
            "pretrained": False,
            "decoder_hidden": getattr(args, "decoder_hidden", 128),
        },
    })


# -----------------------------------------------------------------------------
# Student loading
# -----------------------------------------------------------------------------

def build_texture_student(args) -> nn.Module:
    config = make_student_config(args)
    backbone = getattr(args, "student_backbone", "deit")

    if backbone == "deit":
        model = DeiTTextureClusteringModel(config)
    elif backbone == "mobilenet":
        model = MobileNetTextureClusteringModel(config)
    else:
        raise ValueError(f"Unsupported student_backbone: {backbone}")

    return model


def load_texture_pretrained(student: nn.Module, checkpoint_path: str, strict: bool = False) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)    #### change here weights_only=False

    if isinstance(checkpoint, dict) and "student_state" in checkpoint:
        state_dict = checkpoint["student_state"]
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Clean common wrappers.
    clean_state = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        clean_state[k] = v

    load_info = student.load_state_dict(clean_state, strict=strict)
    logger.info(f"Loaded texture pretrained checkpoint: {checkpoint_path}")
    logger.info(f"Missing keys: {load_info.missing_keys}")
    logger.info(f"Unexpected keys: {load_info.unexpected_keys}")
    return student


# -----------------------------------------------------------------------------
# Feature extractor
# -----------------------------------------------------------------------------

class TextureFeatureExtractor(nn.Module):
    """
    Wraps the texture student and returns features/latent/texture maps.
    """

    def __init__(self, student: nn.Module):
        super().__init__()
        self.student = student

    def forward(self, x: torch.Tensor):
        out = self.student(x, return_features=True)
        return out


# -----------------------------------------------------------------------------
# Classification downstream model
# -----------------------------------------------------------------------------

class TextureClassificationModel(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        num_classes: int,
        freeze_backbone: bool = False,
        classifier_hidden: int = 0,
    ):
        super().__init__()
        self.student = student
        self.num_classes = num_classes

        if freeze_backbone:
            for p in self.student.parameters():
                p.requires_grad = False

        latent_dim = self._infer_latent_dim()

        if classifier_hidden > 0:
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, classifier_hidden),
                nn.BatchNorm1d(classifier_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(classifier_hidden, num_classes),
            )
        else:
            self.classifier = nn.Linear(latent_dim, num_classes)

    def _infer_latent_dim(self) -> int:
        # Most of our students expose latent projection output size through final layer.
        if hasattr(self.student, "latent"):
            for module in reversed(list(self.student.latent.modules())):
                if isinstance(module, nn.Linear):
                    return module.out_features
        raise RuntimeError("Could not infer latent dimension from student.latent")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.student(x, return_features=False)
        logits = self.classifier(out.latent)
        return logits


# # -----------------------------------------------------------------------------
# # Segmentation downstream model
# # -----------------------------------------------------------------------------


class Feature2Pyramid_Scale(nn.Module):
    def __init__(self, embed_dim=192, rescales=[4, 2, 1, 0.5], scale_factor=0.25):
        super().__init__()
        out_channels = int(embed_dim * scale_factor)

        self.projections = nn.ModuleList([
            nn.Conv2d(embed_dim, out_channels, kernel_size=1)
            for _ in rescales
        ])
        self.rescales = rescales

    def forward(self, features):
        outputs = []

        for i, (feat, rescale) in enumerate(zip(features, self.rescales)):
            proj_feat = self.projections[i](feat)

            if rescale != 1.0:
                proj_feat = F.interpolate(
                    proj_feat,
                    scale_factor=rescale,
                    mode="bilinear",
                    align_corners=False,
                )

            outputs.append(proj_feat)

        return outputs


class FPNHead(nn.Module):
    def __init__(
        self,
        in_channels=[48, 48, 48, 48],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        align_corners=False,
    ):
        super().__init__()

        self.in_index = in_index
        self.align_corners = align_corners

        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, kernel_size=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels)):
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        if num_classes == 2:
            self.conv_seg = nn.Conv2d(channels, 1, kernel_size=1)
        else:
            self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None

    def forward(self, features):
        laterals = []

        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(features[self.in_index[i]]))

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        fpn_outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            fpn_outs.append(fpn_conv(laterals[i]))

        target_size = fpn_outs[0].shape[2:]
        fused = fpn_outs[0]

        for i in range(1, len(fpn_outs)):
            upsampled = F.interpolate(
                fpn_outs[i],
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            fused = fused + upsampled

        if self.dropout is not None:
            fused = self.dropout(fused)

        return self.conv_seg(fused)


class TextureSegmentationModel(nn.Module):
    """
    Segmentation model using TinyUSFM-style Feature2Pyramid + FPN decoder.

    It uses your pretrained texture student as the backbone.

    The student currently returns one feature map, so we replicate it four times
    to match the TinyUSFM decoder interface.
    """

    def __init__(
        self,
        student: nn.Module,
        num_classes: int,
        freeze_backbone: bool = False,
        rescales=[4, 2, 1, 0.5],
        scale_factor: float = 0.25,
        fpn_channels: int = 128,
        dropout_ratio: float = 0.1,
        use_texture_fusion: bool = False,     ### ablation with smart fusion
    ):
        super().__init__()

        self.student = student
        self.num_classes = num_classes
        self.use_texture_fusion = use_texture_fusion

        if freeze_backbone:
            for p in self.student.parameters():
                p.requires_grad = False

        self.embed_dim = self._infer_feature_dim()
        neck_channels = int(self.embed_dim * scale_factor)

        self.neck = Feature2Pyramid_Scale(
            embed_dim=self.embed_dim,
            rescales=rescales,
            scale_factor=scale_factor,
        )

        self.decode_head = FPNHead(
            in_channels=[neck_channels, neck_channels, neck_channels, neck_channels],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=fpn_channels,
            dropout_ratio=dropout_ratio,
            num_classes=num_classes,
            align_corners=False,
        )

        if use_texture_fusion:
            dict_size = getattr(self.student, "dict_size", 5)
            self.texture_to_feature = nn.Conv2d(
                dict_size,
                self.embed_dim,
                kernel_size=1,
            )

    def _infer_feature_dim(self) -> int:
        if hasattr(self.student, "embed_dim"):
            return int(self.student.embed_dim)

        if hasattr(self.student, "feature_dim"):
            return int(self.student.feature_dim)

        img_size = getattr(self.student, "img_size", 224)
        was_training = self.student.training
        self.student.eval()

        with torch.no_grad():
            x = torch.randn(1, 1, img_size, img_size)
            out = self.student(x, return_features=True)
            c = out.features.shape[1]

        self.student.train(was_training)
        return c

    def _make_four_features(self, out):
        feat = out.features

        if self.use_texture_fusion:
            tex = F.interpolate(
                out.structure,
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            feat = feat + self.texture_to_feature(tex)

        return [feat, feat, feat, feat]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.student(x, return_features=True)

        features = self._make_four_features(out)

        neck_features = self.neck(features)

        seg_logits = self.decode_head(neck_features)

        seg_logits = F.interpolate(
            seg_logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        return seg_logits
# -----------------------------------------------------------------------------
# Public loader functions
# -----------------------------------------------------------------------------

def load_texture_model_for_classification(args, device):
    pretrained = args.pretrained == "True" if isinstance(args.pretrained, str) else bool(args.pretrained)

    student = build_texture_student(args)
    if pretrained and getattr(args, "checkpoint", None):
        student = load_texture_pretrained(student, args.checkpoint, strict=False)

    model = TextureClassificationModel(
        student=student,
        num_classes=args.num_classes,
        freeze_backbone=getattr(args, "freeze_backbone", "False") == "True",
        classifier_hidden=getattr(args, "classifier_hidden", 0),
    )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: TextureClassificationModel")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    return model


def load_texture_model_for_segmentation(args, device):
    pretrained = args.pretrained == "True" if isinstance(args.pretrained, str) else bool(args.pretrained)

    student = build_texture_student(args)
    if pretrained and getattr(args, "checkpoint", None):
        student = load_texture_pretrained(student, args.checkpoint, strict=False)

    model = TextureSegmentationModel(
        student=student,
        num_classes=args.num_classes,
        freeze_backbone=getattr(args, "freeze_backbone", "False") == "True",
        fpn_channels=getattr(args, "fpn_channels", 128),
    )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: TextureSegmentationModel")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    return model
