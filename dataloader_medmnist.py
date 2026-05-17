"""
MedMNIST Dataset classes for Ark+ cyclic pretraining.
Covers all 12 2D datasets + 6 3D datasets.

Each Dataset returns: (student_view, teacher_view, label)
  - 2D: student/teacher are (C,H,W) float32 tensors
  - 3D: student/teacher are (C,D,H,W) float32 tensors

The student view uses random augmentation; the teacher view uses a
weaker / different random crop — matching the CXR Ark+ pattern.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import random

# ─────────────────────────────────────────────────────────────────────────────
# MedMNIST is required: pip install medmnist
# ─────────────────────────────────────────────────────────────────────────────
import medmnist
from medmnist import INFO

# ─────────────────────────────────────────────────────────────────────────────
# Normalization stats (grayscale → replicate to 3ch for 2D; single ch for 3D)
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MEDMNIST_MEAN_GRAY = [0.5]       # used for 3D (1-channel)
MEDMNIST_STD_GRAY  = [0.5]


# ─────────────────────────────────────────────────────────────────────────────
# 2D MedMNIST Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _make_2d_student_transform(size=28):
    """Strong augmentation for the student view."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def _make_2d_teacher_transform(size=28):
    """Weaker augmentation for the teacher view."""
    return T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(p=0.3),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def _make_2d_val_transform(size=28):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class MedMNIST2DDataset(Dataset):
    """
    Wraps any medmnist 2D flag into a student-teacher dataset.

    Parameters
    ----------
    flag        : medmnist flag string, e.g. 'pathmnist'
    split       : 'train' | 'val' | 'test'
    size        : image resolution (28 by default; use 64/128/224 for bigger)
    mode        : 'train' (dual augmented views) | 'val' | 'test' (single view)
    download    : download dataset if missing
    root        : cache directory (default ~/.medmnist)
    """

    def __init__(self, flag, split='train', size=28,
                 mode='train', download=True, root=None):
        info       = INFO[flag]
        DataClass  = getattr(medmnist, info['python_class'])

        kwargs = dict(split=split, download=download, size=size)
        if root is not None:
            kwargs['root'] = root

        self.dataset   = DataClass(**kwargs)
        self.mode      = mode
        self.task      = info['task']           # 'multi-class' | 'multi-label' | 'binary-class'
        self.n_classes = len(info['label'])
        self.size      = size

        self.student_tf = _make_2d_student_transform(size)
        self.teacher_tf = _make_2d_teacher_transform(size)
        self.val_tf     = _make_2d_val_transform(size)

    # ------------------------------------------------------------------
    def _to_rgb(self, img_np):
        """Convert numpy H×W or H×W×C array to PIL RGB."""
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[-1] == 1:
            img_np = np.concatenate([img_np] * 3, axis=-1)
        return Image.fromarray(img_np.astype(np.uint8))

    def _make_label(self, label_raw):
        """
        medmnist labels are shape (1,) for multi-class/binary,
        or (N,) for multi-label. Return FloatTensor.
        """
        label = label_raw.squeeze()           # scalar or 1-D array
        if self.task == 'multi-label, binary-class':
            return torch.FloatTensor(label.astype(np.float32))
        else:
            # multi-class / binary-class: one-hot
            n = self.n_classes
            lv = int(label) if label.ndim == 0 else int(label[0])
            oh = np.zeros(n, dtype=np.float32)
            oh[lv] = 1.0
            return torch.FloatTensor(oh)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_np, label_raw = self.dataset[idx]      # numpy arrays
        pil = self._to_rgb(img_np)
        label = self._make_label(label_raw)

        if self.mode == 'train':
            student = self.student_tf(pil)
            teacher = self.teacher_tf(pil)
        else:
            v = self.val_tf(pil)
            student = v
            teacher = v

        return student, teacher, label


# ─────────────────────────────────────────────────────────────────────────────
# 3D MedMNIST Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _aug3d_student(volume):
    """
    volume: numpy float32 (D, H, W) in [0,1]
    Returns: torch float32 (1, D, H, W) normalised
    """
    # Random flip along depth and spatial axes
    if random.random() < 0.5:
        volume = volume[::-1].copy()      # flip depth
    if random.random() < 0.5:
        volume = volume[:, ::-1].copy()   # flip H
    if random.random() < 0.5:
        volume = volume[:, :, ::-1].copy() # flip W

    # Random intensity jitter
    alpha = random.uniform(0.8, 1.2)
    beta  = random.uniform(-0.1, 0.1)
    volume = np.clip(volume * alpha + beta, 0.0, 1.0)

    # Normalise
    volume = (volume - 0.5) / 0.5
    return torch.tensor(volume[np.newaxis], dtype=torch.float32)  # (1,D,H,W)


def _aug3d_teacher(volume):
    """Weaker augmentation for teacher view."""
    if random.random() < 0.3:
        volume = volume[::-1].copy()
    volume = (volume - 0.5) / 0.5
    return torch.tensor(volume[np.newaxis], dtype=torch.float32)


def _aug3d_val(volume):
    volume = (volume - 0.5) / 0.5
    return torch.tensor(volume[np.newaxis], dtype=torch.float32)


class MedMNIST3DDataset(Dataset):
    """
    Wraps any medmnist 3D flag into a student-teacher dataset.

    Parameters
    ----------
    flag   : e.g. 'organmnist3d', 'synapsemnist3d', ...
    split  : 'train' | 'val' | 'test'
    mode   : 'train' | 'val' | 'test'
    """

    def __init__(self, flag, split='train',
                 mode='train', download=True, root=None):
        info      = INFO[flag]
        DataClass = getattr(medmnist, info['python_class'])

        kwargs = dict(split=split, download=download)
        if root is not None:
            kwargs['root'] = root

        self.dataset   = DataClass(**kwargs)
        self.mode      = mode
        self.task      = info['task']
        self.n_classes = len(info['label'])

    # ------------------------------------------------------------------
    def _make_label(self, label_raw):
        label = label_raw.squeeze()
        if self.task == 'multi-label, binary-class':
            return torch.FloatTensor(label.astype(np.float32))
        else:
            n  = self.n_classes
            lv = int(label) if label.ndim == 0 else int(label[0])
            oh = np.zeros(n, dtype=np.float32)
            oh[lv] = 1.0
            return torch.FloatTensor(oh)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # medmnist 3D: img shape (D, H, W) or (D, H, W, C); uint8
        img_np, label_raw = self.dataset[idx]

        # Normalise to [0,1]
        volume = img_np.astype(np.float32) / 255.0
        if volume.ndim == 4:          # (D, H, W, C) -> take first channel
            volume = volume[..., 0]

        label = self._make_label(label_raw)

        if self.mode == 'train':
            student = _aug3d_student(volume)
            teacher = _aug3d_teacher(volume)
        else:
            v = _aug3d_val(volume)
            student = v
            teacher = v

        return student, teacher, label


# ─────────────────────────────────────────────────────────────────────────────
# Registry — mirrors dict_dataloarder pattern from the original dataloader.py
# ─────────────────────────────────────────────────────────────────────────────

# 12 × 2D MedMNIST flags
MEDMNIST_2D_FLAGS = [
    'pathmnist',
    'chestmnist',
    'dermamnist',
    'octmnist',
    'pneumoniamnist',
    'retinamnist',
    'breastmnist',
    'bloodmnist',
    'tissuemnist',
    'organamnist',
    'organcmnist',
    'organsmnist',
]

# 6 × 3D MedMNIST flags
MEDMNIST_3D_FLAGS = [
    'organmnist3d',
    'nodulemnist3d',
    'adrenalmnist3d',
    'fracturemnist3d',
    'vesselmnist3d',
    'synapsemnist3d',
]


def build_medmnist_datasets(flag, split, size=28, download=True, root=None):
    """
    Factory that returns (train_ds, val_ds, test_ds) for a given MedMNIST flag.
    Automatically detects 2D vs 3D.
    """
    is_3d = flag in MEDMNIST_3D_FLAGS

    if is_3d:
        train_ds = MedMNIST3DDataset(flag, split='train', mode='train',
                                     download=download, root=root)
        val_ds   = MedMNIST3DDataset(flag, split='val',   mode='val',
                                     download=download, root=root)
        test_ds  = MedMNIST3DDataset(flag, split='test',  mode='test',
                                     download=download, root=root)
    else:
        train_ds = MedMNIST2DDataset(flag, split='train', size=size,
                                     mode='train', download=download, root=root)
        val_ds   = MedMNIST2DDataset(flag, split='val',   size=size,
                                     mode='val',   download=download, root=root)
        test_ds  = MedMNIST2DDataset(flag, split='test',  size=size,
                                     mode='test',  download=download, root=root)

    return train_ds, val_ds, test_ds
