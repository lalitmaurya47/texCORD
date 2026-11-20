"""
Standard-backbone lightweight student models.

This file defines two models that directly plug standard lightweight backbones


1. DeiTTextureClusteringModel
   - backbone: timm `deit_tiny_patch16_224`
   - standard small ViT / transformer baseline

2. MobileNetTextureClusteringModel
   - backbone: torchvision MobileNetV3-Small
   - standard small CNN baseline



    TrainingOutput(
        logits=logits,              # raw K-channel texture logits
        structure=texture_probs,    # K-channel texture probability maps
        reconstruction=xhat,        # reconstructed B-mode image
        latent=latent,              # contrastive embedding
        features=features,          # feature map for USFM KD / texture KD
    )

Install dependencies:

    pip install timm sparsemax torchvision

Usage:

    model = DeiTTextureClusteringModel(config)
    # or
    model = MobileNetTextureClusteringModel(config)

    out = model(x, return_features=True)
    texture_probs = out.structure
    student_features = out.features



"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:
    timm = None

try:
    from sparsemax import Sparsemax
except Exception:
    Sparsemax = None

try:
    from torchvision.models import mobilenet_v3_small
except Exception:
    mobilenet_v3_small = None


@dataclass
class TrainingOutput:
    logits: torch.Tensor
    structure: torch.Tensor
    reconstruction: torch.Tensor
    latent: torch.Tensor
    features: Optional[torch.Tensor] = None


# -----------------------------------------------------------------------------
# Common TUSA-style modules
# -----------------------------------------------------------------------------

class TextureActivation(nn.Module):
    """Sparsemax if installed, otherwise Softmax."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
        self.sparsemax = Sparsemax(dim=dim) if Sparsemax is not None else None

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.sparsemax is not None:
            return self.sparsemax(logits)
        return torch.softmax(logits, dim=self.dim)


class TextureReconstructionHead(nn.Module):
    """
    TUSA-style reconstruction:
    K texture maps -> depthwise texture kernels -> 1x1 B-mode reconstruction.
    """

    def __init__(self, dict_size: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.texture = nn.Conv2d(
            in_channels=dict_size,
            out_channels=dict_size,
            kernel_size=kernel_size,
            groups=dict_size,
            padding=padding,
            bias=False,
        )
        self.to_bmode = nn.Conv2d(dict_size, 1, kernel_size=1)

    def forward(self, structure: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.to_bmode(self.texture(structure)))


class LatentProjection(nn.Module):
    """Projection head used for contrastive learning."""

    def __init__(self, in_dim: int, latent_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(latent_size, latent_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleTextureDecoder(nn.Module):
    """
    Lightweight decoder that upsamples a low-resolution feature map into
    full-resolution K texture logits.
    """

    def __init__(self, in_channels: int, dict_size: int, hidden_channels: int = 128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, dict_size, kernel_size=1),
        )

    def forward(self, features: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


# -----------------------------------------------------------------------------
# 1) DeiT-Tiny TUSA-style model
# -----------------------------------------------------------------------------

class DeiTTextureClusteringModel(nn.Module):
    """
    TUSA-style model using DeiT-Tiny as the student backbone.

    This uses timm's `deit_tiny_patch16_224` with `features_only=False`.
    The transformer token sequence is reshaped into a 14x14 feature map for
    texture decoding and texture-guided distillation.
    """

    def __init__(self, config):
        super().__init__()
        if timm is None:
            raise ImportError("Please install timm: pip install timm")

        self.dict_size = int(config.dict_size)
        self.img_size = int(getattr(config, "img_size", 224))
        self.latent_size = int(getattr(config, "latent_size", 256))

        deit_cfg = getattr(config, "deit", None)
        self.pretrained = bool(getattr(deit_cfg, "pretrained", True)) if deit_cfg is not None else False
        self.backbone_name = getattr(deit_cfg, "backbone", "deit_tiny_patch16_224") if deit_cfg is not None else "deit_tiny_patch16_224"
        decoder_hidden = int(getattr(deit_cfg, "decoder_hidden", 128)) if deit_cfg is not None else 128

        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=self.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=1,
        )

        # DeiT-Tiny has embed_dim=192.
        self.embed_dim = int(getattr(self.backbone, "embed_dim", 192))
        self.patch_size = 16
        self.grid_size = self.img_size // self.patch_size

        self.texture_decoder = SimpleTextureDecoder(
            in_channels=self.embed_dim,
            dict_size=self.dict_size,
            hidden_channels=decoder_hidden,
        )
        self.activation = TextureActivation(dim=1)
        self.bmode = TextureReconstructionHead(self.dict_size, kernel_size=3)
        self.latent = LatentProjection(self.embed_dim, self.latent_size)

    def _tokens_to_feature_map(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert DeiT output tokens to:
          feature_map: [B, C, H, W]
          pooled:      [B, C]
        """
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected DeiT token output [B,N,C], got {tokens.shape}")

        # timm ViT output usually includes cls token if global_pool="".
        patch_tokens = tokens[:, 1:, :] if tokens.shape[1] > self.grid_size * self.grid_size else tokens

        b, n, c = patch_tokens.shape
        side = int(n ** 0.5)
        if side * side != n:
            raise RuntimeError(f"Cannot reshape {n} tokens into square feature map")

        feature_map = patch_tokens.transpose(1, 2).reshape(b, c, side, side)
        pooled = patch_tokens.mean(dim=1)
        return feature_map, pooled

    def forward(self, x: torch.Tensor, return_features: bool = False) -> TrainingOutput:
        tokens = self.backbone.forward_features(x)

        # Some timm versions may return pooled features. Handle both.
        if tokens.ndim == 2:
            raise RuntimeError(
                "Backbone returned pooled features only. Ensure timm create_model uses global_pool='' "
                "so forward_features returns tokens."
            )

        features, pooled = self._tokens_to_feature_map(tokens)
        logits = self.texture_decoder(features, output_size=x.shape[-2:])
        structure = self.activation(logits)
        reconstruction = self.bmode(structure)
        latent = self.latent(pooled)

        return TrainingOutput(
            logits=logits,
            structure=structure,
            reconstruction=reconstruction,
            latent=latent,
            features=features if return_features else None,
        )


# -----------------------------------------------------------------------------
# 2) MobileNetV3-Small TUSA-style model
# -----------------------------------------------------------------------------

class MobileNetTextureClusteringModel(nn.Module):
    """
    TUSA-style model using MobileNetV3-Small as the student backbone.

    The final MobileNet feature map is used for:
      - texture decoding
      - texture-guided distillation
      - contrastive latent projection
    """

    def __init__(self, config):
        super().__init__()
        if mobilenet_v3_small is None:
            raise ImportError("Please install torchvision")

        self.dict_size = int(config.dict_size)
        self.latent_size = int(getattr(config, "latent_size", 256))

        mb_cfg = getattr(config, "mobilenet", None)
        pretrained = bool(getattr(mb_cfg, "pretrained", True)) if mb_cfg is not None else False
        decoder_hidden = int(getattr(mb_cfg, "decoder_hidden", 128)) if mb_cfg is not None else 128

        # torchvision API changed from pretrained= to weights=.
        if pretrained:
            try:
                from torchvision.models import MobileNet_V3_Small_Weights
                weights = MobileNet_V3_Small_Weights.DEFAULT
                base = mobilenet_v3_small(weights=weights)
            except Exception:
                base = mobilenet_v3_small(pretrained=True)
        else:
            try:
                base = mobilenet_v3_small(weights=None)
            except Exception:
                base = mobilenet_v3_small(pretrained=False)

        # Adapt first convolution from 3-channel to 1-channel.
        first_conv = base.features[0][0]
        if first_conv.in_channels != 1:
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                dilation=first_conv.dilation,
                groups=first_conv.groups,
                bias=first_conv.bias is not None,
            )
            with torch.no_grad():
                if first_conv.weight.shape[1] == 3:
                    new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                if first_conv.bias is not None:
                    new_conv.bias.copy_(first_conv.bias)
            base.features[0][0] = new_conv

        self.backbone = base.features
        self.feature_dim = 576  # MobileNetV3-Small final feature channels

        self.texture_decoder = SimpleTextureDecoder(
            in_channels=self.feature_dim,
            dict_size=self.dict_size,
            hidden_channels=decoder_hidden,
        )
        self.activation = TextureActivation(dim=1)
        self.bmode = TextureReconstructionHead(self.dict_size, kernel_size=3)
        self.latent = LatentProjection(self.feature_dim, self.latent_size)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> TrainingOutput:
        features = self.backbone(x)
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)

        logits = self.texture_decoder(features, output_size=x.shape[-2:])
        structure = self.activation(logits)
        reconstruction = self.bmode(structure)
        latent = self.latent(pooled)

        return TrainingOutput(
            logits=logits,
            structure=structure,
            reconstruction=reconstruction,
            latent=latent,
            features=features if return_features else None,
        )


# -----------------------------------------------------------------------------
# Tiny config helper and smoke test
# -----------------------------------------------------------------------------

class AttrDict(dict):
    def __getattr__(self, item):
        value = self[item]
        if isinstance(value, dict):
            return AttrDict(value)
        return value


if __name__ == "__main__":
    config = AttrDict({
        "dict_size": 5,
        "img_size": 224,
        "latent_size": 256,
        "deit": {
            "backbone": "deit_tiny_patch16_224",
            "pretrained": False,
            "decoder_hidden": 128,
        },
        "mobilenet": {
            "pretrained": False,
            "decoder_hidden": 128,
        },
    })

    x = torch.randn(2, 1, 224, 224)

    deit_model = DeiTTextureClusteringModel(config)
    mb_model = MobileNetTextureClusteringModel(config)

    deit_out = deit_model(x, return_features=True)
    mb_out = mb_model(x, return_features=True)

    print("DeiT logits:", deit_out.logits.shape)
    print("DeiT structure:", deit_out.structure.shape)
    print("DeiT reconstruction:", deit_out.reconstruction.shape)
    print("DeiT latent:", deit_out.latent.shape)
    print("DeiT features:", deit_out.features.shape)

    print("MobileNet logits:", mb_out.logits.shape)
    print("MobileNet structure:", mb_out.structure.shape)
    print("MobileNet reconstruction:", mb_out.reconstruction.shape)
    print("MobileNet latent:", mb_out.latent.shape)
    print("MobileNet features:", mb_out.features.shape)
