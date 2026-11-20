"""
downstream_texture_models.py

Downstream wrappers for pretrained TUSA-style student models.

Supports:
  1. Classification:
        pretrained encoder -> global pooling -> classifier

  2. Segmentation:
        pretrained encoder -> multi-scale FPN decoder -> segmentation mask

Compatible checkpoints:
    {
        "student_state": student.state_dict(),
        "projection_state": projection.state_dict(),
        ...
    }

Supported student_backbone:
    - "pvt"
    - "swin"
    - "resnet_unet"
    - "deit"
    - "mobilenet"
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.tgd_v1 import (
        # DeiTTextureClusteringModel,
        # MobileNetTextureClusteringModel,
        SwinTinyTextureClusteringModel,
        PVTv2B0TextureClusteringModel,
        ResNetUNetTextureClusteringModel,
    )
except Exception:
    from models.light_tusa_students import (
        # DeiTTextureClusteringModel,
        # MobileNetTextureClusteringModel,
        SwinTinyTextureClusteringModel,
        PVTv2B0TextureClusteringModel,
        ResNetUNetTextureClusteringModel,
    )


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

class AttrDict(dict):
    def __getattr__(self, item):
        value = self[item]
        if isinstance(value, dict):
            return AttrDict(value)
        return value


def str2bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in ["true", "1", "yes", "y"]
    return bool(x)


def make_student_config(args):
    """
    Build config expected by the SSL student models.
    """

    return AttrDict({
        "dict_size": getattr(args, "dict_size", 5),
        "img_size": getattr(args, "img_size", 224),
        "latent_size": getattr(args, "latent_size", 256),
        "student_backbone": getattr(args, "student_backbone", "pvt"),

        # "deit": {
        #     "backbone": getattr(args, "deit_backbone", "deit_tiny_patch16_224"),
        #     "pretrained": False,
        #     "decoder_hidden": getattr(args, "decoder_hidden", 128),
        # },

        # "mobilenet": {
        #     "pretrained": False,
        #     "decoder_hidden": getattr(args, "decoder_hidden", 128),
        # },

        "swin": {
            "backbone": getattr(args, "swin_backbone", "swin_tiny_patch4_window7_224"),
            "pretrained": False,
            "decoder_channels": getattr(args, "decoder_channels", 128),
        },

        "pvt": {
            "backbone": getattr(args, "pvt_backbone", "pvt_v2_b1"),
            "pretrained": False,
            "decoder_channels": getattr(args, "decoder_channels", 96),
        },

        "resnet": {
            "pretrained": False,
            "decoder_channels": getattr(args, "decoder_channels", 128),
        },
    })


# -----------------------------------------------------------------------------
# Build and load pretrained student
# -----------------------------------------------------------------------------

def build_texture_student(args) -> nn.Module:
    config = make_student_config(args)
    backbone = getattr(args, "student_backbone", "pvt")

    if backbone == "pvt":
        return PVTv2B0TextureClusteringModel(config)

    if backbone == "swin":
        return SwinTinyTextureClusteringModel(config)

    if backbone == "resnet_unet":
        return ResNetUNetTextureClusteringModel(config)

    # if backbone == "deit":
    #     return DeiTTextureClusteringModel(config)

    # if backbone == "mobilenet":
    #     return MobileNetTextureClusteringModel(config)

    raise ValueError(f"Unsupported student_backbone: {backbone}")


def load_texture_pretrained(
    student: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "student_state" in checkpoint:
        state_dict = checkpoint["student_state"]
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

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
# Encoder-only feature extractor
# -----------------------------------------------------------------------------

class TextureEncoder(nn.Module):
    """
    Extracts encoder features only from a pretrained TUSA-style student.

    For segmentation:
        returns multi-scale features:
            [B, C1, H/4,  W/4]
            [B, C2, H/8,  W/8]
            [B, C3, H/16, W/16]
            [B, C4, H/32, W/32]

    For classification:
        final feature map is globally pooled.
    """

    def __init__(self, student: nn.Module):
        super().__init__()
        self.student = student
        self.backbone_type = self._infer_backbone_type()

    def _infer_backbone_type(self) -> str:
        name = self.student.__class__.__name__.lower()

        if "pvt" in name:
            return "pvt"

        if "swin" in name:
            return "swin"

        if "resnet" in name:
            return "resnet_unet"

        if "deit" in name:
            return "deit"

        if "mobilenet" in name:
            return "mobilenet"

        raise ValueError(f"Could not infer backbone type from {self.student.__class__.__name__}")

    @staticmethod
    def _to_nchw(feat: torch.Tensor, channels: Optional[int] = None) -> torch.Tensor:
        """
        timm hierarchical transformer features may be NCHW or NHWC.
        Convert to NCHW.
        """

        if feat.ndim != 4:
            raise RuntimeError(f"Expected 4D feature map, got {feat.shape}")

        if channels is not None:
            if feat.shape[1] == channels:
                return feat

            if feat.shape[-1] == channels:
                return feat.permute(0, 3, 1, 2).contiguous()

        # fallback heuristic
        if feat.shape[1] <= 2048:
            return feat

        return feat.permute(0, 3, 1, 2).contiguous()

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.backbone_type in ["pvt", "swin"]:
            feats = self.student.backbone(x)
            channels = list(self.student.backbone.feature_info.channels())

            feats = [
                self._to_nchw(f, c)
                for f, c in zip(feats, channels)
            ]

            return feats

        if self.backbone_type == "resnet_unet":
            s0 = self.student.stem(x)          # 112x112, 64
            x0 = self.student.maxpool(s0)      # 56x56, 64

            s1 = self.student.layer1(x0)       # 56x56, 64
            s2 = self.student.layer2(s1)       # 28x28, 128
            s3 = self.student.layer3(s2)       # 14x14, 256
            s4 = self.student.layer4(s3)       # 7x7, 512

            return [s1, s2, s3, s4]

        if self.backbone_type == "deit":
            tokens = self.student.backbone.forward_features(x)

            if tokens.ndim != 3:
                raise RuntimeError(f"Expected DeiT tokens [B, N, C], got {tokens.shape}")

            grid_size = self.student.grid_size
            patch_tokens = (
                tokens[:, 1:, :]
                if tokens.shape[1] > grid_size * grid_size
                else tokens
            )

            b, n, c = patch_tokens.shape
            side = int(n ** 0.5)

            feat = patch_tokens.transpose(1, 2).reshape(b, c, side, side)

            # single feature map fallback
            return [feat]

        if self.backbone_type == "mobilenet":
            feat = self.student.backbone(x)

            # single feature map fallback
            return [feat]

        raise ValueError(f"Unsupported backbone type: {self.backbone_type}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.forward_features(x)

    def get_feature_channels(self, img_size: int = 224) -> List[int]:
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        x = torch.randn(1, 1, img_size, img_size, device=device)

        with torch.no_grad():
            feats = self.forward_features(x)

        self.train(was_training)

        return [f.shape[1] for f in feats]


# -----------------------------------------------------------------------------
# Classification model
# -----------------------------------------------------------------------------

class TextureClassificationModel(nn.Module):
    """
    Encoder-only classification model.

    Uses:
        pretrained encoder -> final feature map -> global average pooling -> classifier
    """

    def __init__(
        self,
        student: nn.Module,
        num_classes: int,
        freeze_backbone: bool = False,
        classifier_hidden: int = 0,
        img_size: int = 224,
    ):
        super().__init__()

        self.encoder = TextureEncoder(student)
        self.num_classes = num_classes

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        feature_channels = self.encoder.get_feature_channels(img_size=img_size)
        final_dim = feature_channels[-1]

        if classifier_hidden > 0:
            self.classifier = nn.Sequential(
                nn.Linear(final_dim, classifier_hidden),
                nn.BatchNorm1d(classifier_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(classifier_hidden, num_classes),
            )
        else:
            self.classifier = nn.Linear(final_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        final_feat = feats[-1]

        pooled = F.adaptive_avg_pool2d(final_feat, 1).flatten(1)
        logits = self.classifier(pooled)

        return logits


# -----------------------------------------------------------------------------
# Segmentation decoder
# -----------------------------------------------------------------------------

class FPNHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        channels: int = 128,
        dropout_ratio: float = 0.1,
        num_classes: int = 2,
        align_corners: bool = False,
    ):
        super().__init__()

        self.align_corners = align_corners

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        if num_classes == 2:
            self.conv_seg = nn.Conv2d(channels, 1, kernel_size=1)
        else:
            self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        laterals = []

        for feat, lateral_conv in zip(features, self.lateral_convs):
            laterals.append(lateral_conv(feat))

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        fpn_outs = []

        for feat, fpn_conv in zip(laterals, self.fpn_convs):
            fpn_outs.append(fpn_conv(feat))

        target_size = fpn_outs[0].shape[-2:]
        fused = fpn_outs[0]

        for i in range(1, len(fpn_outs)):
            fused = fused + F.interpolate(
                fpn_outs[i],
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        fused = self.dropout(fused)

        return self.conv_seg(fused)


class SingleFeaturePyramid(nn.Module):
    """
    Fallback for DeiT/MobileNet, which may expose only one encoder feature map.

    Converts one feature map into a 4-level pyramid.
    """

    def __init__(self, in_channels: int, out_channels: int = 128):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

    def forward(self, feat: torch.Tensor) -> List[torch.Tensor]:
        feat = self.proj(feat)

        p1 = F.interpolate(feat, scale_factor=4.0, mode="bilinear", align_corners=False)
        p2 = F.interpolate(feat, scale_factor=2.0, mode="bilinear", align_corners=False)
        p3 = feat
        p4 = F.interpolate(feat, scale_factor=0.5, mode="bilinear", align_corners=False)

        pyramids = [p1, p2, p3, p4]
        pyramids = [refine(p) for refine, p in zip(self.refine, pyramids)]

        return pyramids


class TextureSegmentationModel(nn.Module):
    """
    Encoder-only segmentation model.

    For PVT/Swin/ResNet-UNet:
        uses true multi-scale encoder features.

    For DeiT/MobileNet:
        falls back to a synthetic pyramid from a single feature map.
    """

    def __init__(
        self,
        student: nn.Module,
        num_classes: int,
        freeze_backbone: bool = False,
        fpn_channels: int = 128,
        dropout_ratio: float = 0.1,
        img_size: int = 224,
    ):
        super().__init__()

        self.encoder = TextureEncoder(student)
        self.num_classes = num_classes
        self.img_size = img_size

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        feature_channels = self.encoder.get_feature_channels(img_size=img_size)

        self.uses_single_feature = len(feature_channels) == 1

        if self.uses_single_feature:
            self.single_pyramid = SingleFeaturePyramid(
                in_channels=feature_channels[0],
                out_channels=fpn_channels,
            )
            decoder_in_channels = [
                fpn_channels,
                fpn_channels,
                fpn_channels,
                fpn_channels,
            ]
        else:
            self.single_pyramid = None
            decoder_in_channels = feature_channels

        self.decode_head = FPNHead(
            in_channels=decoder_in_channels,
            channels=fpn_channels,
            dropout_ratio=dropout_ratio,
            num_classes=num_classes,
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        if self.uses_single_feature:
            features = self.single_pyramid(features[0])

        seg_logits = self.decode_head(features)

        seg_logits = F.interpolate(
            seg_logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return seg_logits


# -----------------------------------------------------------------------------
# Public loader functions
# -----------------------------------------------------------------------------

def load_texture_model_for_classification(args, device):
    pretrained = str2bool(getattr(args, "pretrained", False))

    student = build_texture_student(args)

    if pretrained and getattr(args, "checkpoint", None):
        student = load_texture_pretrained(
            student,
            args.checkpoint,
            strict=False,
        )

    model = TextureClassificationModel(
        student=student,
        num_classes=args.num_classes,
        freeze_backbone=str2bool(getattr(args, "freeze_backbone", False)),
        classifier_hidden=int(getattr(args, "classifier_hidden", 0)),
        img_size=int(getattr(args, "img_size", 224)),
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Model loaded: TextureClassificationModel")
    logger.info(f"Backbone: {getattr(args, 'student_backbone', 'pvt')}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model


def load_texture_model_for_segmentation(args, device):
    pretrained = str2bool(getattr(args, "pretrained", False))

    student = build_texture_student(args)

    if pretrained and getattr(args, "checkpoint", None):
        student = load_texture_pretrained(
            student,
            args.checkpoint,
            strict=False,
        )

    model = TextureSegmentationModel(
        student=student,
        num_classes=args.num_classes,
        freeze_backbone=str2bool(getattr(args, "freeze_backbone", False)),
        fpn_channels=int(getattr(args, "fpn_channels", 128)),
        dropout_ratio=float(getattr(args, "dropout_ratio", 0.1)),
        img_size=int(getattr(args, "img_size", 224)),
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Model loaded: TextureSegmentationModel")
    logger.info(f"Backbone: {getattr(args, 'student_backbone', 'pvt')}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model