"""
trainer_medmnist.py
───────────────────
Ark+ training loop adapted for MedMNIST 2D+3D.

Changes vs the original trainer.py
────────────────────────────────────
1.  wandb import is now *optional* (hard import crashed when wandb
    wasn't installed / initialised).
2.  save_image is skipped gracefully for 3D volumes (can't write a
    3D tensor as a jpeg in the middle of training).
3.  Evaluation loop accepts both 4-D (B,C,H,W) and 5-D (B,C,D,H,W)
    batches — no TenCrop augmentation is used for 3D test sets.
4.  test_classification supports both 4-D and 5-D inputs.
"""

import time
import torch
from tqdm import tqdm
import numpy as np

from utils import MetricLogger, ProgressLogger

# ── optional wandb ──────────────────────────────────────────────────────────
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _log_wandb(key, value):
    if _WANDB_AVAILABLE:
        try:
            wandb.log({key: value})
        except Exception:
            pass   # silently ignore if run not initialised


def _save_image_safe(tensor_np, path):
    """Save first 2D slice only; silently skip 3D volumes."""
    try:
        from utils import save_image
        if tensor_np.ndim == 3:          # (C, H, W)
            img = tensor_np.transpose(1, 2, 0)  # → (H, W, C)
        elif tensor_np.ndim == 4:        # (C, D, H, W) — 3D volume
            return                       # skip
        else:
            return
        save_image(img, path)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, use_head_n, dataset_name,
                    data_loader_train, device, criterion,
                    optimizer, epoch,
                    ema_mode, teacher, momentum_schedule, it):
    """
    One full pass over data_loader_train.

    Returns: updated iteration counter `it`.
    """
    batch_time  = MetricLogger('Time',                    ':6.3f')
    losses_cls  = MetricLogger(f'Loss_{dataset_name}_cls', ':.4e')
    losses_mse  = MetricLogger(f'Loss_{dataset_name}_mse', ':.4e')
    progress    = ProgressLogger(
        len(data_loader_train),
        [batch_time, losses_cls, losses_mse],
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()
    MSE  = torch.nn.MSELoss()
    coff = (momentum_schedule[it] - 0.9) * 5   # consistency loss weight
    end  = time.time()

    for i, (samples1, samples2, targets) in enumerate(data_loader_train):
        samples1 = samples1.float().to(device)
        samples2 = samples2.float().to(device)
        # targets: long index for multi-class, float for multi-label/binary
        targets  = targets.to(device)

        feat_t, pred_t = teacher(samples2, use_head_n)
        feat_s, pred_s = model(samples1,   use_head_n)

        loss_cls   = criterion(pred_s, targets)
        loss_const = MSE(feat_s, feat_t)
        loss       = (1 - coff) * loss_cls + coff * loss_const

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_cls.update(loss_cls.item(), samples1.size(0))
        losses_mse.update(loss_const.item(), samples1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            progress.display(i)
            # Save sample images (2D only — safe helper skips 3D)
            _save_image_safe(samples1[0].cpu().numpy(),
                             f"Models/student_{dataset_name}_{i}")
            _save_image_safe(samples2[0].cpu().numpy(),
                             f"Models/teacher_{dataset_name}_{i}")

        if ema_mode == 'iteration':
            ema_update_teacher(model, teacher, momentum_schedule, it)
            it += 1

    if ema_mode == 'epoch':
        ema_update_teacher(model, teacher, momentum_schedule, it)
        it += 1

    _log_wandb(f"train_loss_cls_{dataset_name}",  losses_cls.avg)
    _log_wandb(f"train_loss_mse_{dataset_name}",  losses_mse.avg)

    return it   # ← caller must update its own `it` counter


# ─────────────────────────────────────────────────────────────────────────────

def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = momentum_schedule[it]
        for param_q, param_k in zip(model.parameters(),
                                    teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, use_head_n, data_loader_val, device, criterion, dataset_name):
    model.eval()
    batch_time = MetricLogger('Time',   ':6.3f')
    losses     = MetricLogger('Loss',   ':.4e')
    progress   = ProgressLogger(
        len(data_loader_val),
        [batch_time, losses],
        prefix=f'Val_{dataset_name}: ',
    )

    end = time.time()
    with torch.no_grad():
        for i, (samples, _, targets) in enumerate(data_loader_val):
            samples = samples.float().to(device)
            targets = targets.to(device)  # long for multi-class, float for others

            _, outputs = model(samples, use_head_n)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), samples.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

    return losses.avg


# ─────────────────────────────────────────────────────────────────────────────

def test_classification(model, use_head_n, data_loader_test,
                        device, multiclass=False):
    """
    Runs inference over data_loader_test.

    Supports:
        4-D input (B, C, H, W)  → standard 2D single/TenCrop
        5-D input (B, C, D, H, W) → 3D volume (no TenCrop)

    Returns (y_test, p_test) as FloatTensors on `device`.
    """
    model.eval()
    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)

    with torch.no_grad():
        for i, (samples, _, targets) in enumerate(tqdm(data_loader_test)):
            targets = targets.to(device)
            # y_test must be float for metric_AUROC; long labels need casting
            y_test  = torch.cat((y_test, targets.float()), 0)

            ndim = samples.dim()

            if ndim == 4:
                # (B, C, H, W)  — plain single view
                bs = samples.size(0)
                varInput = samples.to(device)
                _, out = model(varInput, use_head_n)
                outMean = out

            elif ndim == 5:
                # Could be (B, n_crops, C, H, W) [TenCrop] or (B, C, D, H, W) [3D]
                # Distinguish: MedMNIST 3D dataset returns (B, C, D, H, W)
                # TenCrop returns (B, n_crops, C, H, W) where n_crops is typically 10
                # We check: if dim-1 is small (≤10) and dim-2 looks like channels ≤3
                #           → TenCrop; else → 3D volume
                d1 = samples.size(1)
                d2 = samples.size(2)
                if d1 <= 10 and d2 <= 3:
                    # TenCrop path
                    bs, n_crops, c, h, w = samples.size()
                    varInput = samples.view(-1, c, h, w).to(device)
                    _, out   = model(varInput, use_head_n)
                    out      = out.view(bs, n_crops, -1).mean(1)
                    outMean  = out
                else:
                    # 3D volume (B, C, D, H, W)
                    bs = samples.size(0)
                    varInput = samples.to(device)
                    _, out = model(varInput, use_head_n)
                    outMean = out
            else:
                raise ValueError(f"Unexpected input shape: {samples.shape}")

            if multiclass:
                outMean = torch.softmax(outMean, dim=1)
            else:
                outMean = torch.sigmoid(outMean)

            p_test = torch.cat((p_test, outMean.data), 0)

    return y_test, p_test
