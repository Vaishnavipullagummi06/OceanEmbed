"""
OceanEmbed Model (SIH26066)
U-Net encoder ("ocean embedding") and depth reconstruction decoder head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEPTHS, VARS


def ConvBlock(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


class OceanEmbed(nn.Module):
    """
    U-Net architecture mapping surface inputs to subsurface temperatures at N depths.
    Includes automatic spatial padding to ensure feature map sizes are multiples of 4.
    """
    def __init__(self, cin=len(VARS), emb=64, nd=len(DEPTHS)):
        super().__init__()
        self.cin = cin
        self.emb = emb
        self.nd = nd

        # Encoder
        self.e1 = ConvBlock(cin, 32)
        self.e2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 64))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, emb))

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d2 = ConvBlock(emb + 64, 64)
        self.d1 = ConvBlock(64 + 32, 32)
        self.head = nn.Conv2d(32, nd, kernel_size=1)

    def _pad_input(self, x):
        """Pad spatial dimensions to nearest multiples of 4."""
        h, w = x.shape[-2:]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, h, w

    def _unpad_output(self, out, original_h, original_w):
        """Crop output back to original spatial dimensions."""
        return out[..., :original_h, :original_w]

    def embed(self, x):
        """Extract bottleneck ocean embedding representations."""
        x, _, _ = self._pad_input(x)
        return self.e3(self.e2(self.e1(x)))

    def forward(self, x):
        x_padded, orig_h, orig_w = self._pad_input(x)
        
        # Encoder forward pass
        a = self.e1(x_padded)
        b = self.e2(a)
        c = self.e3(b)
        
        # Decoder forward pass with skip connections
        d2_in = torch.cat([self.up(c), b], dim=1)
        d2_out = self.d2(d2_in)
        
        d1_in = torch.cat([self.up(d2_out), a], dim=1)
        d1_out = self.d1(d1_in)
        
        out_padded = self.head(d1_out)
        return self._unpad_output(out_padded, orig_h, orig_w)


def masked_mse_loss(pred, target, mask=None):
    """
    Computes Mean Squared Error, ignoring land pixels if mask is provided.
    mask: float tensor of shape (B, 1, H, W) or (B, H, W) where 1=ocean, 0=land.
    """
    if mask is None:
        return F.mse_loss(pred, target)
    
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    
    # Broadcast mask across depth dimension
    diff_sq = (pred - target) ** 2 * mask
    total_valid = mask.sum() * pred.shape[1] + 1e-8
    return diff_sq.sum() / total_valid
