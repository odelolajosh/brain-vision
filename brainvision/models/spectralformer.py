"""
SpectralFormer: Hyperspectral Image Classification with Vision Transformer.
Hong et al. (2021) — "SpectralFormer: Rethinking Hyperspectral Image
Classification with Transformers"
IEEE TGRS. https://doi.org/10.1109/TGRS.2021.3130716

Contains the full transformer stack:
  Residual, PreNorm, FeedForward, Attention, Transformer, ViT, SpectralFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ── Transformer building blocks ──────────────────────────────────────────────


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1]
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# ── Transformer encoder ─────────────────────────────────────────────────────


class Transformer(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode
    ):
        super().__init__()
        self.mode = mode
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout))
                        ),
                    ]
                )
            )

        # CAF skip connections — only used in CAF mode
        self.skipcat = nn.ModuleList(
            [
                nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0)
                for _ in range(depth - 2)
            ]
        )

    def forward(self, x, mask=None):
        if self.mode == "ViT":
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)

        elif self.mode == "CAF":
            last_output = []
            for nl, (attn, ff) in enumerate(self.layers):
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](
                        torch.cat(
                            [x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3
                        )
                    ).squeeze(3)
                x = attn(x, mask=mask)
                x = ff(x)
        return x


# ── ViT backbone ─────────────────────────────────────────────────────────────


class ViT(nn.Module):
    """
    Internal ViT backbone used by SpectralFormer.
    Not intended for direct use — use SpectralFormer instead.
    """

    def __init__(
        self,
        image_size,
        near_band,
        num_patches,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=16,
        dropout=0.0,
        emb_dropout=0.0,
        mode="ViT",
    ):
        super().__init__()
        patch_dim = image_size**2 * near_band

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode
        )
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x, mask=None):
        x = self.patch_to_embedding(x)  # (B, N, dim)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_latent(x[:, 0])  # CLS token
        return self.mlp_head(x)


# ── SpectralFormer (public API) ──────────────────────────────────────────────


class SpectralFormer(nn.Module):
    """
    SpectralFormer: Hyperspectral Image Classification with Vision Transformer.
    Hong et al. (2021) — https://doi.org/10.1109/TGRS.2021.3130716

    Treats the spectrum as a sequence of grouped band tokens.
    Each token covers near_band adjacent spectral bands (via unfold).

    Input:  (B, input_channels)  — pixel spectrum from HSIPixelDataset
    Output: (B, n_classes)       — raw logits

    Args:
        input_channels : number of spectral bands after preprocessing (128)
        n_classes      : number of tissue classes (4)
        near_band      : number of adjacent bands per token (default 3)
        dim            : transformer embedding dimension (default 64)
        depth          : number of transformer layers (default 5)
        heads          : number of attention heads (default 4)
        dim_head       : dimension per attention head (default 16)
        mlp_dim        : feedforward hidden dimension (default 8)
        dropout        : attention + FFN dropout (default 0.1)
        emb_dropout    : embedding dropout (default 0.1)
        mode           : 'ViT' (standard) or 'CAF' (cross-layer adaptive fusion)
    """

    def __init__(
        self,
        input_channels: int = 128,
        n_classes: int = 4,
        near_band: int = 3,
        dim: int = 64,
        depth: int = 5,
        heads: int = 4,
        dim_head: int = 16,
        mlp_dim: int = 8,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        mode: str = "ViT",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.near_band = near_band

        self.vit = ViT(
            image_size=1,  # pixel-level — no spatial patch
            near_band=near_band,  # bands per token
            num_patches=input_channels,  # one token per band position
            num_classes=n_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            mode=mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C) — pixel spectrum

        Builds grouped band tokens via unfold:
          each position i gets a token of near_band consecutive bands
          centred on band i — (B, C, near_band)
        """
        pad = self.near_band // 2
        x_pad = F.pad(x, (pad, pad), mode="reflect")  # (B, C + 2*pad)
        x_seq = x_pad.unfold(1, self.near_band, 1)  # (B, C, near_band)
        return self.vit(x_seq)
