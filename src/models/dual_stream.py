"""
Dual-Stream Fusion Model for UCI HAR:
  Stream 1: PatchTST on raw 1D time series (128, 9)
  Stream 2: GAF + 2D-CNN on polar encoding images (128, 128, 18)
  → Concatenated features → FC classifier
"""

import torch
import torch.nn as nn
import math

from transformers import PatchTSTConfig, PatchTSTModel
from src.models.gaf import GAFEncoder


# ─────────────────────────────────────────────
#  Stream 1: PatchTST (1D sequence encoder)
# ─────────────────────────────────────────────
class PatchTSTStream(nn.Module):
    """
    Extract a fixed-size feature vector from PatchTST's [CLS] token output.
    """
    def __init__(self, num_channels: int, d_model: int = 128,
                 num_layers: int = 3, num_heads: int = 8,
                 ffn_dim: int = 256, patch_len: int = 16,
                 stride: int = 8, context_len: int = 128,
                 dropout: float = 0.1):
        super().__init__()

        config = PatchTSTConfig(
            num_input_channels=num_channels,
            context_length=context_len,
            patch_length=patch_len,
            stride=stride,
            num_hidden_layers=num_layers,
            d_model=d_model,
            num_attention_heads=num_heads,
            ffn_dim=ffn_dim,
            norm_type="batchnorm",
            pre_norm=True,
            positional_encoding_type="sincos",
            use_cls_token=True,
            attention_dropout=dropout,
            ff_dropout=dropout,
        )
        self.backbone = PatchTSTModel(config)
        self.feat_dim = d_model  # [CLS] token dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)  ← PatchTST expects (batch, seq_len, channels)
        Returns:
            (B, d_model)  ← mean-pooled over patches
        """
        out = self.backbone(past_values=x)
        # out.last_hidden_state: (B, num_channels, num_patches, d_model)
        # Pool over num_patches (dim=2) → (B, num_channels, d_model)
        # Then mean-pool over num_channels (dim=1) → (B, d_model)
        feat = out.last_hidden_state.mean(dim=2)        # (B, C, d_model)
        feat = feat.mean(dim=1)                         # (B, d_model)
        return feat


# ─────────────────────────────────────────────
#  Stream 2: GAF + 2D-CNN
# ─────────────────────────────────────────────
class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bn: bool = True,
                 relu: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class GAFCNNStream(nn.Module):
    """
    Encode each of the 9 channels as GASF+GADF → 18-channel image (T×T).
    Apply 2D-CNN → AdaptiveAvgPool → fixed-size feature vector.
    Works with any sequence length T >= 8.
    """
    def __init__(self, in_channels: int = 9, feat_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()

        self.gaf = GAFEncoder(field_type="BOTH")   # → (B, C*2, T, T)
        gaf_channels = in_channels * 2             # GASF + GADF per channel

        self.conv1 = ConvBlock2D(gaf_channels,  32,  kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout2d(dropout))

        self.conv2 = ConvBlock2D(32,  64,  kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.Sequential(nn.MaxPool2d(2), nn.Dropout2d(dropout))

        self.conv3 = ConvBlock2D(64,  128, kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.Dropout2d(dropout)            # no pooling here

        self.conv4 = ConvBlock2D(128, 256, kernel_size=3, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # any spatial >=1 → 1×1
        self.feat_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw signal  ← from Dataset
        Returns:
            (B, 256) global feature vector
        """
        gaf = self.gaf(x)                             # (B, 18, T, T)
        x = self.conv1(gaf);  x = self.pool1(x)       # T/2
        x = self.conv2(x);     x = self.pool2(x)      # T/4
        x = self.conv3(x);     x = self.pool3(x)      # T/8 (→ >=1 via adaptive)
        x = self.conv4(x)                               # T/8 / 2 → >=1
        x = self.pool(x)                                # → 1×1 always
        feat = x.view(x.size(0), -1)                  # (B, 256)
        return feat


# ─────────────────────────────────────────────
#  Multi-Head Attention Fusion
# ─────────────────────────────────────────────
class AttentionFusion(nn.Module):
    """
    Multi-head self-attention over two feature streams.
    Treats the two stream features as a 2-element sequence, allowing
    cross-stream interaction before final classification.
    Uses stochastic depth (drop path) for stronger regularization.
    """
    def __init__(self, feat_dim: int, num_heads: int = 4,
                 dropout: float = 0.1, attn_dropout: float = 0.5,
                 drop_path_rate: float = 0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = feat_dim // num_heads
        self.drop_path_rate = drop_path_rate
        assert feat_dim % num_heads == 0, f"{feat_dim} not divisible by {num_heads}"

        self.qkv_proj  = nn.Linear(feat_dim, 3 * feat_dim)
        self.out_proj  = nn.Linear(feat_dim, feat_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, feat_dim)  — concatenated features from two streams
        Returns:
            (B, feat_dim)    — attended features
        """
        B, D = x.shape
        seq = x.unsqueeze(1)                       # (B, 1, D)
        seq = seq.expand(-1, 2, -1)                # (B, 2, D)

        # Self-attention
        qkv  = self.qkv_proj(seq)
        qkv  = qkv.reshape(B, 2, 3, self.num_heads, self.head_dim)
        qkv  = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out  = (attn @ v).transpose(1, 2)
        out  = out.reshape(B, 2, D)
        out  = self.out_proj(out)
        out  = self.proj_drop(out)

        pooled = out.mean(dim=1)

        # Stochastic depth: randomly skip the attention residual
        if self.training:
            if torch.rand(1).item() < self.drop_path_rate:
                pooled = torch.zeros_like(pooled)

        return self.norm(x + pooled)                # (B, D)


# ─────────────────────────────────────────────
#  Dual-Stream Fusion
# ─────────────────────────────────────────────
class DualStreamHAR(nn.Module):
    """
    Two-stream architecture:
      Raw signal → [Stream 1: PatchTST]  → (B, 128)
      Raw signal → [Stream 2: GAF+CNN]   → (B, 256)
      Concatenate → Multi-Head Attention → FC → 6-class softmax
    """
    def __init__(self, num_classes: int = 6, num_channels: int = 9,
                 seq_len: int = 128,
                 d_model: int = 128, cnn_feat: int = 256,
                 dropout: float = 0.4):
        super().__init__()

        self.stream1 = PatchTSTStream(
            num_channels=num_channels,
            d_model=d_model,
            num_layers=3,
            num_heads=8,
            ffn_dim=256,
            patch_len=16,
            stride=8,
            context_len=seq_len,
            dropout=dropout,
        )
        self.stream2 = GAFCNNStream(
            in_channels=num_channels,
            feat_dim=cnn_feat,
            dropout=dropout,
        )

        self.fusion_dim = self.stream1.feat_dim + self.stream2.feat_dim  # 128 + 256

        self.attention_fusion = AttentionFusion(
            feat_dim=self.fusion_dim,
            num_heads=4,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) = (B, 9, 128) raw time series
        Returns:
            (B, num_classes) logits
        """
        x_t = x.transpose(1, 2)           # (B, T, C) for PatchTST
        f1 = self.stream1(x_t)           # (B, 128)
        f2 = self.stream2(x)              # (B, 256)
        fused = torch.cat([f1, f2], dim=-1)  # (B, 384)
        fused = self.attention_fusion(fused)  # (B, 384) — cross-stream attention
        return self.classifier(fused)     # (B, 6)


if __name__ == "__main__":
    model = DualStreamHAR(num_classes=6, num_channels=9, seq_len=128)
    x = torch.randn(8, 9, 128)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  |  Trainable: {trainable:,}")

    out = model(x)
    print(f"Output shape: {out.shape}  (should be (8, 6))")
    assert out.shape == (8, 6)
    print("Dual-stream model OK!")
