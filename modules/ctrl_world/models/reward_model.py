from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def make_resnet18_backbone(in_ch: int = 3, pretrained: bool = True):
    """
    Returns ResNet-18 backbone (up to avgpool) adapted to in_ch channels.
    If in_ch != 3, replace conv1 and init extra channels as mean of RGB weights.
    """
    resnet = models.resnet18(pretrained=pretrained)

    if in_ch == 3:
        modules = list(resnet.children())[:-1]
        backbone = nn.Sequential(*modules)
        return backbone

    # replace conv1
    old_conv = resnet.conv1  # shape (64,3,7,7)
    old_weight = old_conv.weight.data  # (64,3,7,7)

    new_conv = nn.Conv2d(
        in_ch,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        bias=(old_conv.bias is not None),
    )

    with torch.no_grad():
        if in_ch > 3:
            mean_w = old_weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
            new_w = torch.zeros(
                (old_weight.size(0), in_ch, *old_weight.shape[2:]),
                dtype=old_weight.dtype,
                device=old_weight.device,
            )
            new_w[:, :3, :, :] = old_weight
            for i in range(3, in_ch):
                new_w[:, i : i + 1, :, :] = mean_w
            new_conv.weight.data.copy_(new_w)
        else:
            # in_ch < 3
            new_w = old_weight[:, :in_ch, :, :].clone()
            new_conv.weight.data.copy_(new_w)
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

    resnet.conv1 = new_conv
    modules = list(resnet.children())[:-1]
    backbone = nn.Sequential(*modules)
    return backbone


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        # use batch_first=True for (B, L, D) API
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        q: (B, Lq, D)
        kv: (B, Lk, D)
        returns: out: (B, Lq, D)
        """
        attn_out, _ = self.attn(query=q, key=kv, value=kv, attn_mask=attn_mask)
        q = q + attn_out
        q = self.ln1(q)
        ff = self.ffn(q)
        out = q + ff
        out = self.ln2(out)
        return out


class VisionActionClassifier(nn.Module):
    """
    Vision-Action classifier using cross-attention.
    Inputs:
      - images: (B_T, C, H, W)
      - actions: (B_T, action_dim)  (already encoded)
    Outputs:
      - logits: (B_T, num_classes)
      - probs:  (B_T, num_classes)
    """

    def __init__(
        self,
        latent_dim: int = 512,
        action_dim: int = 1024,
        num_classes: int = 2,
        num_action_tokens: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        freeze_vision: bool = True,
        in_channels: int = 3,  # input image channels
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_action_tokens = num_action_tokens
        self.action_dim = action_dim

        # Vision backbone (ResNet18 up to avgpool), supports in_channels != 3
        self.vision_feature_extractor = make_resnet18_backbone(
            in_ch=in_channels, pretrained=pretrained_backbone
        )
        if freeze_vision:
            for p in self.vision_feature_extractor.parameters():
                p.requires_grad = False

        vision_out_dim = 512  # resnet18 final conv output dim
        if latent_dim != vision_out_dim:
            self.vision_proj = nn.Linear(vision_out_dim, latent_dim)
        else:
            self.vision_proj = nn.Identity()

        # Action encoder (project encoded action vector to tokens)
        # action_dim assumed large (1024). Project to num_action_tokens * latent_dim
        self.action_proj = nn.Linear(action_dim, latent_dim * num_action_tokens)

        self.action_ln = nn.LayerNorm(latent_dim)

        # Cross-attention block: vision token (Lq=1) queries action tokens (Lk=num_action_tokens)
        self.cross_block = CrossAttentionBlock(
            embed_dim=latent_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, num_classes),
        )

    def forward(self, images: torch.Tensor, actions: torch.Tensor):
        """
        images: (B_T, C, H, W)
        actions: (B_T, action_dim)
        returns: logits (B_T, num_classes), probs (B_T, num_classes)
        """
        B_T = images.shape[0]
        # vision backbone -> (B_T, 512, 1, 1)
        v = self.vision_feature_extractor(images)  # (B_T, 512, 1, 1)
        v = v.view(B_T, -1)  # (B_T, 512)
        v = self.vision_proj(v)  # (B_T, latent_dim)

        # action -> tokens
        a = self.action_proj(actions)  # (B_T, latent_dim * num_action_tokens)
        a = a.view(B_T, self.num_action_tokens, self.latent_dim)  # (B_T, Lk, D)
        a = self.action_ln(a)

        # prepare for cross-attn: q (B_T, 1, D), kv (B_T, Lk, D)
        q = v.unsqueeze(1)  # (B_T, 1, D)
        kv = a  # (B_T, Lk, D)

        out = self.cross_block(q, kv)  # (B_T, 1, D)
        out = out.squeeze(1)  # (B_T, D)

        logits = self.classifier(out)  # (B_T, num_classes)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def predict_score(self, images: torch.Tensor, actions: torch.Tensor):
        logits, probs = self.forward(images, actions)
        assert self.num_classes == 2, "num_classes must be 2"
        if self.num_classes == 1:
            return torch.sigmoid(logits.view(-1))
        elif self.num_classes == 2:
            return probs[:, 1]
        else:
            return probs.max(dim=-1)[0]
