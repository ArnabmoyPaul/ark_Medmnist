"""
engine_medmnist.py
──────────────────
Ark+ cyclic pretraining engine for all 18 MedMNIST datasets (2D + 3D).

Key differences from the original engine.py
─────────────────────────────────────────────
1.  Uses build_omni_model_medmnist (supports unified 2D/3D backbone).
2.  Passes dataset_dims so the engine knows which loss / metric to apply.
3.  Multi-class datasets use CrossEntropyLoss; binary/multi-label use
    BCEWithLogitsLoss — exactly as the original.
4.  trainer_medmnist functions are used (wandb-optional, 3D-safe).
5.  test_classification AUC computation handles multi-class via accuracy
    (like the original) and multi-label via AUROC.
"""

import os
import copy
import sys
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from models_medmnist import build_omni_model_medmnist, save_checkpoint
from utils import metric_AUROC, cosine_scheduler
from trainer_medmnist import train_one_epoch, test_classification, evaluate

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

sys.setrecursionlimit(40000)


# ─────────────────────────────────────────────────────────────────────────────

def _criterion_for(task_type):
    """Return the appropriate loss for a dataset's task type."""
    if task_type == 'multi-class classification':
        return torch.nn.CrossEntropyLoss()
    else:   # binary classification | multi-label classification
        return torch.nn.BCEWithLogitsLoss()


def _is_multiclass(task_type):
    return task_type == 'multi-class classification'


# ─────────────────────────────────────────────────────────────────────────────

def omni_engine_medmnist(args,
                         model_path, output_path,
                         dataset_list,
                         datasets_config,
                         dataset_train_list,
                         dataset_val_list,
                         dataset_test_list):
    """
    Main entry point.  Mirror of the original omni_engine(), adapted for
    the MedMNIST 18-dataset setting.

    Parameters
    ──────────
    dataset_list        : list of dataset name strings matching datasets_config
    datasets_config     : dict parsed from datasets_config_medmnist.yaml
    dataset_{train,val,test}_list : list of torch Datasets (one per dataset)
    """
    device = torch.device(args.device)
    cudnn.benchmark = True

    # ── Logging paths ────────────────────────────────────────────────────────
    exp = 'Ark_Plus_MedMNIST'
    for ds in dataset_list:
        exp += '_' + ds
    run_model_path  = os.path.join(model_path, exp, args.exp_name)
    os.makedirs(run_model_path, exist_ok=True)
    os.makedirs(output_path,    exist_ok=True)

    log_file    = os.path.join(run_model_path, 'train.log')
    output_file = os.path.join(output_path,    f'{exp}_{args.exp_name}_results.txt')

    # ── DataLoaders ──────────────────────────────────────────────────────────
    # Cap workers: Colab/low-CPU machines warn (and slow down) above 2
    import multiprocessing
    max_workers = min(args.workers, multiprocessing.cpu_count(), 2)
    print(f"Using {max_workers} DataLoader workers.")

    loaders_train = [
        DataLoader(d, batch_size=args.batch_size, shuffle=True,
                   num_workers=max_workers, pin_memory=True)
        for d in dataset_train_list
    ]
    loaders_val = [
        DataLoader(d, batch_size=args.batch_size, shuffle=False,
                   num_workers=max_workers, pin_memory=True)
        for d in dataset_val_list
    ]
    loaders_test = [
        DataLoader(d, batch_size=max(1, args.batch_size // 2), shuffle=False,
                   num_workers=max_workers, pin_memory=True)
        for d in dataset_test_list
    ]

    num_classes_list = [
        len(datasets_config[ds]['diseases']) for ds in dataset_list
    ]
    print("num_classes_list:", num_classes_list)

    # ── Build student & teacher ───────────────────────────────────────────────
    model   = build_omni_model_medmnist(args, num_classes_list)
    teacher = build_omni_model_medmnist(args, num_classes_list)

    if torch.cuda.device_count() > 1:
        model   = torch.nn.DataParallel(model)
        teacher = torch.nn.DataParallel(teacher)

    model.to(device)
    teacher.to(device)

    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Student and Teacher built: {args.model_name}")

    # ── EMA momentum schedule ─────────────────────────────────────────────────
    if args.ema_mode == 'epoch':
        momentum_schedule = cosine_scheduler(
            args.momentum_teacher, 1,
            args.pretrain_epochs, len(dataset_list),
        )
    else:   # iteration
        iters_per_epoch = sum(len(dl) for dl in loaders_train)
        momentum_schedule = cosine_scheduler(
            args.momentum_teacher, 1,
            args.pretrain_epochs, iters_per_epoch,
        )

    # ── Optimiser & LR scheduler ─────────────────────────────────────────────
    optimizer    = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = float('inf')
    save_stem   = os.path.join(run_model_path, exp)

    if args.mode == 'train' and args.resume:
        ckpt_path = save_stem + '.pth.tar'
        if os.path.isfile(ckpt_path):
            print(f"=> Loading checkpoint '{ckpt_path}'")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            start_epoch = ckpt['epoch'] + 1
            best_val    = ckpt.get('lossMIN', float('inf'))

            sd = ckpt['state_dict']
            # Optionally drop task heads so they are re-initialised
            if getattr(args, 'reinit_heads', False):
                sd = {k: v for k, v in sd.items()
                      if not k.startswith('omni_heads.')}

            model.load_state_dict(sd, strict=False)
            teacher.load_state_dict(ckpt['teacher'], strict=False)
            lr_scheduler.load_state_dict(ckpt['scheduler'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f"=> Resumed from epoch {start_epoch - 1}")
        else:
            print(f"=> No checkpoint found at '{ckpt_path}'")

    # ── Log args ─────────────────────────────────────────────────────────────
    with open(log_file, 'a') as f:
        f.write(str(args) + '\n')

    if args.mode != 'train':
        return   # evaluation-only path not implemented here; extend as needed

    # ── Training loop ─────────────────────────────────────────────────────────
    test_results         = []
    test_results_teacher = []
    it = start_epoch * len(dataset_list)

    for epoch in range(start_epoch, args.pretrain_epochs):

        # ── One cyclic pass over all datasets ────────────────────────────
        for i, loader_tr in enumerate(loaders_train):
            task_type = datasets_config[dataset_list[i]]['task_type']
            criterion = _criterion_for(task_type)
            it = train_one_epoch(
                model, i, dataset_list[i],
                loader_tr, device, criterion,
                optimizer, epoch,
                args.ema_mode, teacher, momentum_schedule, it,
            )

        # ── Validation ───────────────────────────────────────────────────
        val_losses = []
        for i, loader_v in enumerate(loaders_val):
            task_type = datasets_config[dataset_list[i]]['task_type']
            criterion = _criterion_for(task_type)
            vl = evaluate(model, i, loader_v, device, criterion, dataset_list[i])
            val_losses.append(vl)

        avg_val = np.mean(val_losses)

        # Which metric to watch for LR scheduling
        if args.val_loss_metric == 'average':
            watch_metric = avg_val
        elif args.val_loss_metric in dataset_list:
            watch_metric = val_losses[dataset_list.index(args.val_loss_metric)]
        else:
            watch_metric = avg_val

        lr_scheduler.step(watch_metric)

        print(f"Epoch {epoch:04d}: avg_val_loss={avg_val:.5f}")

        # ── Save latest checkpoint ────────────────────────────────────────
        ckpt_state = {
            'epoch':      epoch,
            'lossMIN':    val_losses,
            'state_dict': model.state_dict(),
            'teacher':    teacher.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'scheduler':  lr_scheduler.state_dict(),
        }
        save_checkpoint(ckpt_state, filename=save_stem)

        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch:04d}: avg_val_loss={avg_val:.5f}\n")
            f.write(f"  Datasets  : {dataset_list}\n")
            f.write(f"  Val Losses: {val_losses}\n")

        # ── Periodic test evaluation ──────────────────────────────────────
        if epoch % args.test_epoch == 0 or epoch + 1 == args.pretrain_epochs:
            save_checkpoint(ckpt_state,
                            filename=save_stem + str(epoch))

            t_res, t_res_teacher = [], []

            with open(output_file, 'a') as writer:
                writer.write(f"Epoch {epoch:04d}:\n")

                for i, ds_name in enumerate(dataset_list):
                    task_type  = datasets_config[ds_name]['task_type']
                    diseases   = datasets_config[ds_name]['diseases']
                    multiclass = _is_multiclass(task_type)

                    writer.write(f"{ds_name} Val Loss = {val_losses[i]:.5f}\n")
                    print(f">> {ds_name} diseases = {diseases}")

                    y_s, p_s = test_classification(
                        model,   i, loaders_test[i], device, multiclass)
                    y_t, p_t = test_classification(
                        teacher, i, loaders_test[i], device, multiclass)

                    if multiclass:
                        # y_s is float cast of long index → shape (B,) or (B,1)
                        y_s_idx = y_s.cpu().numpy().flatten().astype(int)
                        y_t_idx = y_t.cpu().numpy().flatten().astype(int)
                        p_s_idx = np.argmax(p_s.cpu().numpy(), axis=1)
                        p_t_idx = np.argmax(p_t.cpu().numpy(), axis=1)
                        acc_s = accuracy_score(y_s_idx, p_s_idx)
                        acc_t = accuracy_score(y_t_idx, p_t_idx)
                        print(f">> {ds_name}: Student ACC={acc_s:.4f}, "
                              f"Teacher ACC={acc_t:.4f}")
                        writer.write(
                            f"{ds_name}: Student ACC={acc_s:.4f}, "
                            f"Teacher ACC={acc_t:.4f}\n")
                        t_res.append(acc_s)
                        t_res_teacher.append(acc_t)

                    else:
                        # AUROC for binary / multi-label
                        n_cls = len(diseases)
                        auc_s = metric_AUROC(y_s, p_s, n_cls)
                        auc_t = metric_AUROC(y_t, p_t, n_cls)

                        m_auc_s = np.mean(auc_s) if auc_s else 0.0
                        m_auc_t = np.mean(auc_t) if auc_t else 0.0

                        print(f">> {ds_name}: Student mAUC={m_auc_s:.4f}, "
                              f"Teacher mAUC={m_auc_t:.4f}")
                        writer.write(
                            f"{ds_name}: Student mAUC={m_auc_s:.4f}, "
                            f"Teacher mAUC={m_auc_t:.4f}\n"
                            f"  Individual AUC (S): {np.round(auc_s, 4).tolist()}\n"
                            f"  Individual AUC (T): {np.round(auc_t, 4).tolist()}\n")
                        t_res.append(m_auc_s)
                        t_res_teacher.append(m_auc_t)

            test_results.append(t_res)
            test_results_teacher.append(t_res_teacher)
            print(f"[Test Summary]\n  Student:  {t_res}\n  Teacher: {t_res_teacher}")

    # ── Final summary ─────────────────────────────────────────────────────────
    with open(output_file, 'a') as writer:
        writer.write(
            f"\n{'='*60}\n"
            f"FINAL — Student  metrics:\n{test_results}\n"
            f"FINAL — Teacher  metrics:\n{test_results_teacher}\n"
        )
