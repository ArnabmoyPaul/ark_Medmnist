"""
models_medmnist.py
──────────────────
Extends the original models.py with a 3D-aware build path.

For 2D datasets  → ArkSwinTransformer  (same as original Ark+)
For 3D datasets  → ArkPlus3D           (from ark_plus_model.py)

build_omni_model_medmnist(args, num_classes_list, dataset_dims)
  dataset_dims : list of int (2 or 3), one per dataset in dataset_list
                 e.g. [2, 2, 3, 2, 3] for a mixed run

Mixed 2D+3D sessions use a *single* shared Swin backbone:
  - 3D inputs are rearranged to (B*D, C, H, W) before the 2D backbone
    (same strategy as ArkPlus3D.forward_features)
  - This keeps the architecture unified and avoids maintaining two
    separate student-teacher pairs.
"""

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

import timm.models.swin_transformer as swin
from timm.models.helpers import load_state_dict
from convnext import ConvNeXt
from utils import remap_pretrained_keys_swin
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# Re-use ArkSwinTransformer from original models.py, with one addition:
# forward() can handle 5-D (B,C,D,H,W) inputs by folding depth into batch.
# ─────────────────────────────────────────────────────────────────────────────

class ArkSwinTransformer(swin.SwinTransformer):
    """
    Unified 2D/3D Ark+ backbone.

    For 3D inputs (B, C, D, H, W):
        1. Rearrange to (B*D, C, H, W)
        2. Run through 2D Swin
        3. Average-pool over the depth dimension → (B, F)
    This matches ArkPlus3D.forward_features behaviour and lets a single
    model handle all 18 MedMNIST datasets.
    """

    def __init__(self, num_classes_list,
                 projector_features=None, use_mlp=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None

        # Optional projection head (consistency loss)
        self.projector = None
        if projector_features:
            enc_f = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(
                    nn.Linear(enc_f, self.num_features),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.num_features, self.num_features),
                )
            else:
                self.projector = nn.Linear(enc_f, self.num_features)

        # One linear head per dataset
        self.omni_heads = nn.ModuleList([
            nn.Linear(self.num_features, nc) if nc > 0 else nn.Identity()
            for nc in num_classes_list
        ])

    # ------------------------------------------------------------------
    def _extract_features(self, x):
        """
        x : (B, C, H, W) or (B, C, D, H, W)
        Returns (B, F) feature vector.
        """
        if x.dim() == 5:
            B, C, D, H, W = x.shape
            x = rearrange(x, 'b c d h w -> (b d) c h w')
            feats = super().forward_features(x)          # (B*D, F)
            feats = feats.view(B, D, -1).mean(dim=1)     # (B, F)
        else:
            feats = super().forward_features(x)          # (B, F)
        return feats

    def forward(self, x, head_n=None):
        feats = self._extract_features(x)
        if self.projector:
            feats = self.projector(feats)
        if head_n is not None:
            return feats, self.omni_heads[head_n](feats)
        return [head(feats) for head in self.omni_heads]

    def generate_embeddings(self, x, after_proj=True):
        feats = self._extract_features(x)
        if after_proj and self.projector:
            feats = self.projector(feats)
        return feats


# ─────────────────────────────────────────────────────────────────────────────
# ArkConvNeXt  (unchanged from original, kept for completeness)
# ─────────────────────────────────────────────────────────────────────────────

class ArkConvNeXt(ConvNeXt):
    def __init__(self, num_classes_list,
                 projector_features=None, use_mlp=False,
                 encoder_features=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None

        self.projector = None
        if projector_features:
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(
                    nn.Linear(encoder_features, self.num_features),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.num_features, self.num_features),
                )
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

        self.omni_heads = nn.ModuleList([
            nn.Linear(self.num_features, nc) if nc > 0 else nn.Identity()
            for nc in num_classes_list
        ])

    def forward(self, x, head_n=None):
        feats = self.forward_features(x)
        if self.projector:
            feats = self.projector(feats)
        if head_n is not None:
            return feats, self.omni_heads[head_n](feats)
        return [head(feats) for head in self.omni_heads]

    def generate_embeddings(self, x, after_proj=True):
        feats = self.forward_features(x)
        if after_proj and self.projector:
            feats = self.projector(feats)
        return feats


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_omni_model_medmnist(args, num_classes_list):
    """
    Builds the student (or teacher) model for MedMNIST Ark+ pretraining.

    The same unified ArkSwinTransformer handles 2D inputs natively and
    3D inputs via depth-folding — no separate 3D model needed.

    Supported --model values:
        swin_tiny   : embed_dim=96,  depths=(2,2,6,2),   heads=(3,6,12,24)
        swin_small  : embed_dim=96,  depths=(2,2,18,2),  heads=(3,6,12,24)
        swin_base   : embed_dim=128, depths=(2,2,18,2),  heads=(4,8,16,32)
        swin_large  : embed_dim=192, depths=(2,2,18,2),  heads=(6,12,24,48)
        conv_base   : ConvNeXt-Base
    """
    model_name = args.model_name
    pf         = args.projector_features
    use_mlp    = args.use_mlp

    # MedMNIST default resolution is 28×28; backbone img_size must match.
    img_size   = getattr(args, 'crop_size', 28)

    if model_name == 'swin_tiny':
        model = ArkSwinTransformer(
            num_classes_list, pf, use_mlp,
            img_size=img_size, patch_size=2, window_size=7,
            embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
        )
    elif model_name == 'swin_small':
        model = ArkSwinTransformer(
            num_classes_list, pf, use_mlp,
            img_size=img_size, patch_size=2, window_size=7,
            embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24),
        )
    elif model_name == 'swin_base':
        model = ArkSwinTransformer(
            num_classes_list, pf, use_mlp,
            img_size=img_size, patch_size=4, window_size=7,
            embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        )
    elif model_name == 'swin_large':
        model = ArkSwinTransformer(
            num_classes_list, pf, use_mlp,
            img_size=img_size, patch_size=4, window_size=7,
            embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        )
    elif model_name == 'conv_base':
        model = ArkConvNeXt(
            num_classes_list, pf, use_mlp,
            depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         "Choose from: swin_tiny | swin_small | swin_base | swin_large | conv_base")

    # ── Load pretrained weights ──────────────────────────────────────────
    if getattr(args, 'pretrained_weights', None) is not None:
        pw = args.pretrained_weights
        if pw.startswith('https'):
            state_dict = load_state_dict_from_url(url=pw, map_location='cpu')
        else:
            state_dict = load_state_dict(pw)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # Strip attention masks (always re-initialised)
        k_del = [k for k in state_dict if 'attn_mask' in k]
        for k in k_del:
            del state_dict[k]
        if k_del:
            print(f"[build_omni_model] Removed keys: {k_del}")

        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[build_omni_model] Loaded pretrained weights — {msg}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helper  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state, filename='model'):
    torch.save(state, filename + '.pth.tar')
