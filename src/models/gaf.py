"""
Gramian Angular Field (GAF) transformation for time series.
Supports GASF (Gramian Angular Summation Field) and GADF (Gramian Angular Difference Field).
"""

import numpy as np
import torch
import torch.nn as nn


class GAFEncoder(nn.Module):
    """
    Encode a batch of multivariate time series (B, C, T) into GAF images.

    For each channel independently:
      1. Min-max normalize to [0, 1]
      2. Convert to polar coordinates (angle = acos(value), radius = t/N)
      3. Compute Gramian matrix:
         - GASF: cos(angle_i + angle_j)  → captures positive temporal correlation
         - GADF: sin(angle_i - angle_j)  → captures negative/oscillatory patterns

    GASF and GADF are stacked as two channels → output (B, C*2, N, N)
    """
    def __init__(self, field_type: str = "GASF"):
        super().__init__()
        self.field_type = field_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw time series, any scale
        Returns:
            gaf: (B, C*2, T, T) GASF + GADF stacked per channel
                 If field_type=='GASF' → (B, C, T, T)
                 If field_type=='BOTH'  → (B, C*2, T, T)
        """
        B, C, T = x.shape
        device = x.device

        # Step 1: Min-max normalize per channel
        x_min = x.min(dim=2, keepdim=True)[0]          # (B, C, 1)
        x_max = x.max(dim=2, keepdim=True)[0]          # (B, C, 1)
        denom = x_max - x_min
        denom = torch.where(denom < 1e-8, torch.ones_like(denom), denom)
        xn = (x - x_min) / denom                      # (B, C, T), values in [0,1]

        # Step 2: Polar coordinates — scaled radius to spread time across [0,1]
        phi = torch.acos(torch.clamp(xn, -1.0, 1.0))  # (B, C, T), angle ∈ [0, π]
        r = torch.linspace(0, 1, T, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        r = r.expand(B, 1, T)                          # (B, 1, T)

        # Step 3: Compute GAF per channel
        # phi.unsqueeze(-1): (B, C, T, 1)  — for row index
        # phi.unsqueeze(-2): (B, C, 1, T)  — for col index
        phi_col = phi.unsqueeze(-1)                   # (B, C, T, 1)
        phi_row = phi.unsqueeze(-2)                   # (B, C, 1, T)

        gasf = torch.cos(phi_row + phi_col)          # (B, C, T, T)
        gadf = torch.sin(phi_row - phi_col)          # (B, C, T, T)

        if self.field_type == "GASF":
            return gasf
        elif self.field_type == "GADF":
            return gadf
        else:  # "BOTH"
            return torch.cat([gasf, gadf], dim=1)    # (B, C*2, T, T)


class TrivializedGAF(nn.Module):
    """
    Trivialized GAF: applies the centering trick to reduce diagonal dominance.
    G' = G - t·I/2, where t = trace(G) / N
    This makes the GAF less dominated by the time×time cosine self-interaction.
    """
    def __init__(self, field_type: str = "BOTH"):
        super().__init__()
        self.field_type = field_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        device = x.device

        x_min = x.min(dim=2, keepdim=True)[0]
        x_max = x.max(dim=2, keepdim=True)[0]
        denom = x_max - x_min
        denom = torch.where(denom < 1e-8, torch.ones_like(denom), denom)
        xn = (x - x_min) / denom

        phi = torch.acos(torch.clamp(xn, -1.0, 1.0))
        phi_col = phi.unsqueeze(-1)
        phi_row = phi.unsqueeze(-2)

        gasf = torch.cos(phi_row + phi_col)
        gadf = torch.sin(phi_row - phi_col)

        # Trivialization: subtract mean diagonal per batch/channel
        eye = torch.eye(T, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        t_gasf = (gasf * (1 - eye)).sum(dim=(-1, -2), keepdim=True) / (T * T - T)
        t_gadf = (gadf * (1 - eye)).sum(dim=(-1, -2), keepdim=True) / (T * T - T)
        gasf = gasf - t_gasf.unsqueeze(-1).unsqueeze(-1) * eye
        gadf = gadf - t_gadf.unsqueeze(-1).unsqueeze(-1) * eye

        if self.field_type == "GASF":
            return gasf
        elif self.field_type == "GADF":
            return gadf
        else:
            return torch.cat([gasf, gadf], dim=1)


def gaf_batch(x: torch.Tensor, field_type: str = "BOTH",
              trivialized: bool = False) -> torch.Tensor:
    """
    Functional wrapper for one-off GAF computation.
    Args:
        x: (B, C, T) or (C, T)  — auto-detects batch dimension
        field_type: "GASF" | "GADF" | "BOTH"
        trivialized: if True, apply trivialization
    Returns:
        gaf: (B, C*2, T, T) or (C*2, T, T)
    """
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze = True

    encoder = TrivializedGAF(field_type) if trivialized else GAFEncoder(field_type)
    out = encoder(x)

    if squeeze:
        out = out.squeeze(0)
    return out


if __name__ == "__main__":
    # Quick sanity check
    x = torch.randn(4, 9, 128)
    encoder = GAFEncoder(field_type="BOTH")
    gaf = encoder(x)
    print(f"Input:  {x.shape}")
    print(f"GAF:    {gaf.shape}  (B, C*2, T, T)")
    assert gaf.shape == (4, 18, 128, 128)

    trig = TrivializedGAF(field_type="BOTH")
    tgaf = trig(x)
    print(f"T-GAF:  {tgaf.shape}")

    # Verify GASF diagonal = 1 (cos(0+0)=1)
    gasf_only = GAFEncoder(field_type="GASF")
    g = gasf_only(x)
    diag = torch.diagonal(g, dim1=2, dim2=3)  # (B, C)
    print(f"GASF diagonal mean: {diag.mean():.4f}  (should be ~1.0)")
    print("GAF module OK!")
