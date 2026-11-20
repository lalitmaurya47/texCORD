"""
Dense lightweight TexCORD student models.


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

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
    from torchvision.models import resnet18
except Exception:
    resnet18 = None


# -----------------------------------------------------------------------------
# Output dataclass
# -----------------------------------------------------------------------------

@dataclass
class TrainingOutput:
    logits: torch.Tensor
    structure: torch.Tensor
    reconstruction: torch.Tensor
    latent: torch.Tensor
    features: Optional[torch.Tensor] = None


# -----------------------------------------------------------------------------
# Common TexCORD modules
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
    Texture reconstruction:
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


class ConvBNAct(nn.Module):
    """Small convolution block for dense decoder."""

    def __init__(self, in_channels: int, out_channels: int, act: str = "gelu"):
        super().__init__()
        if act == "relu":
            activation = nn.ReLU(inplace=True)
        else:
            activation = nn.GELU()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    U-Net/FPN upsampling block:
    upsample low-res feature -> concatenate skip -> refine.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.refine = ConvBNAct(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)




class PVTv2B0TextureClusteringModel(nn.Module):
    """
    Lightweight transformer TexCORD model using PVTv2-B0.

    
      - hierarchical features
      - lower params/FLOPs
      - suitable for segmentation/texture maps
    """

    def __init__(self, config):
        super().__init__()
        if timm is None:
            raise ImportError("Please install timm: pip install timm")

        self.dict_size = int(config.dict_size)
        self.img_size = int(getattr(config, "img_size", 224))
        self.latent_size = int(getattr(config, "latent_size", 256))

        pvt_cfg = getattr(config, "pvt", None)
        self.backbone_name = (
            getattr(pvt_cfg, "backbone", "pvt_v2_b1")
            if pvt_cfg is not None
            else "pvt_v2_b1"
        )
        self.pretrained = (
            bool(getattr(pvt_cfg, "pretrained", True))
            if pvt_cfg is not None
            else True
        )
        decoder_channels = (
            int(getattr(pvt_cfg, "decoder_channels", 96))
            if pvt_cfg is not None
            else 96
        )

        self.backbone = timm.create_model(
            self.backbone_name,
            pretrained=self.pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=1,
        )

        self.feature_channels = list(self.backbone.feature_info.channels())
        c1, c2, c3, c4 = self.feature_channels

        self.up3 = UpBlock(c4, c3, decoder_channels)
        self.up2 = UpBlock(decoder_channels, c2, decoder_channels)
        self.up1 = UpBlock(decoder_channels, c1, decoder_channels)

        self.texture_head = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, self.dict_size, kernel_size=1),
        )

        self.activation = TextureActivation(dim=1)
        self.bmode = TextureReconstructionHead(self.dict_size, kernel_size=3)
        self.latent = LatentProjection(c4, self.latent_size)

    @staticmethod
    def _to_nchw(feat: torch.Tensor, channels: int) -> torch.Tensor:
        if feat.ndim != 4:
            raise RuntimeError(f"Expected 4D feature map, got {feat.shape}")

        if feat.shape[1] == channels:
            return feat

        if feat.shape[-1] == channels:
            return feat.permute(0, 3, 1, 2).contiguous()

        raise RuntimeError(
            f"Cannot infer feature format for shape {feat.shape} with channels={channels}"
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> TrainingOutput:
        feats = self.backbone(x)
        feats = [
            self._to_nchw(f, c)
            for f, c in zip(feats, self.feature_channels)
        ]

        f1, f2, f3, f4 = feats

        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)

        logits = self.texture_head(d1)
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        structure = self.activation(logits)
        reconstruction = self.bmode(structure)

        pooled = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        latent = self.latent(pooled)

        return TrainingOutput(
            logits=logits,
            structure=structure,
            reconstruction=reconstruction,
            latent=latent,
            features=d1 if return_features else None,
        )

        

# -----------------------------------------------------------------------------
# 2) Lightweight ResNet18 U-Net CNN model
# -----------------------------------------------------------------------------

class ResNetUNetTextureClusteringModel(nn.Module):
    """
    CNN TexCORD model using ResNet18 encoder + U-Net decoder.

    This is stronger for dense texture prediction than using only the final
    MobileNet/ResNet feature map because it restores spatial detail using skips.
    """

    def __init__(self, config):
        super().__init__()
        if resnet18 is None:
            raise ImportError("Please install torchvision")

        self.dict_size = int(config.dict_size)
        self.latent_size = int(getattr(config, "latent_size", 256))

        resnet_cfg = getattr(config, "resnet", None)
        pretrained = (
            bool(getattr(resnet_cfg, "pretrained", True))
            if resnet_cfg is not None
            else True
        )
        decoder_channels = (
            int(getattr(resnet_cfg, "decoder_channels", 128))
            if resnet_cfg is not None
            else 128
        )

        if pretrained:
            try:
                from torchvision.models import ResNet18_Weights
                base = resnet18(weights=ResNet18_Weights.DEFAULT)
            except Exception:
                base = resnet18(pretrained=True)
        else:
            try:
                base = resnet18(weights=None)
            except Exception:
                base = resnet18(pretrained=False)

        # Adapt first convolution from RGB to single-channel ultrasound.
        old_conv = base.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            if old_conv.weight.shape[1] == 3:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        base.conv1 = new_conv

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
        )
        self.maxpool = base.maxpool

        self.layer1 = base.layer1  # 56x56, 64
        self.layer2 = base.layer2  # 28x28, 128
        self.layer3 = base.layer3  # 14x14, 256
        self.layer4 = base.layer4  # 7x7, 512

        self.up3 = UpBlock(512, 256, decoder_channels)
        self.up2 = UpBlock(decoder_channels, 128, decoder_channels)
        self.up1 = UpBlock(decoder_channels, 64, decoder_channels)
        self.up0 = UpBlock(decoder_channels, 64, decoder_channels)

        self.texture_head = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels, act="relu"),
            nn.Conv2d(decoder_channels, self.dict_size, kernel_size=1),
        )

        self.activation = TextureActivation(dim=1)
        self.bmode = TextureReconstructionHead(self.dict_size, kernel_size=3)
        self.latent = LatentProjection(512, self.latent_size)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> TrainingOutput:
        s0 = self.stem(x)          # 112x112, 64
        x0 = self.maxpool(s0)      # 56x56, 64

        s1 = self.layer1(x0)       # 56x56, 64
        s2 = self.layer2(s1)       # 28x28, 128
        s3 = self.layer3(s2)       # 14x14, 256
        s4 = self.layer4(s3)       # 7x7, 512

        d3 = self.up3(s4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        logits = self.texture_head(d0)
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        structure = self.activation(logits)
        reconstruction = self.bmode(structure)

        pooled = F.adaptive_avg_pool2d(s4, 1).flatten(1)
        latent = self.latent(pooled)

        return TrainingOutput(
            logits=logits,
            structure=structure,
            reconstruction=reconstruction,
            latent=latent,
            features=d0 if return_features else None,
        )


# -----------------------------------------------------------------------------
# Model statistics: parameters, size, GFLOPs
# -----------------------------------------------------------------------------

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """
    Approximate state_dict size in MB.
    Includes parameters and buffers.
    """
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)


def calculate_gflops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 1, 224, 224),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Optional[float]:
    """
    Returns GFLOPs for one forward pass.

    First tries fvcore.
    If fvcore is unavailable, tries thop.
    If both are unavailable, returns None.
    """
    model = model.to(device)
    model.eval()

    dummy = torch.randn(*input_size).to(device)

    # fvcore method
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            flops = FlopCountAnalysis(model, dummy).total()
        return flops / 1e9
    except Exception as e:
        print(f"[GFLOPs] fvcore failed: {e}")

    # thop method
    try:
        from thop import profile
        with torch.no_grad():
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception as e:
        print(f"[GFLOPs] thop failed: {e}")

    return None


def print_model_summary(
    model: nn.Module,
    name: str,
    input_size: Tuple[int, int, int, int] = (1, 1, 224, 224),
):
    total_params = count_total_parameters(model)
    trainable_params = count_trainable_parameters(model)
    size = model_size_mb(model)
    gflops = calculate_gflops(model, input_size=input_size)

    print("=" * 80)
    print(name)
    print("=" * 80)
    print(f"Total parameters:     {total_params / 1e6:.3f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.3f} M")
    print(f"Model size:           {size:.2f} MB")
    if gflops is not None:
        print(f"GFLOPs:               {gflops:.3f}")
    else:
        print("GFLOPs:               Could not calculate. Install fvcore or thop.")
    print("=" * 80)


# -----------------------------------------------------------------------------
# Config helper and smoke test
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

        "resnet": {
            "pretrained": False,
            "decoder_channels": 128,
        },
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 1, 224, 224).to(device)

    swin_model = SwinTinyTextureClusteringModel(config).to(device)
    resnet_unet_model = ResNetUNetTextureClusteringModel(config).to(device)

    with torch.no_grad():
        swin_out = swin_model(x, return_features=True)
        resnet_out = resnet_unet_model(x, return_features=True)

    print("\nSwin-Tiny output shapes")
    print("logits:", swin_out.logits.shape)
    print("structure:", swin_out.structure.shape)
    print("reconstruction:", swin_out.reconstruction.shape)
    print("latent:", swin_out.latent.shape)
    print("features:", swin_out.features.shape)

    print("\nResNet18-UNet output shapes")
    print("logits:", resnet_out.logits.shape)
    print("structure:", resnet_out.structure.shape)
    print("reconstruction:", resnet_out.reconstruction.shape)
    print("latent:", resnet_out.latent.shape)
    print("features:", resnet_out.features.shape)

   

    print_model_summary(
        resnet_unet_model,
        "ResNet18-UNet TexCORD CNN student",
        input_size=(1, 1, 224, 224),
    )