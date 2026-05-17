"""
main_medmnist.py
────────────────
Entry point for Ark+ cyclic pretraining on all 18 MedMNIST datasets.

Quick-start examples
────────────────────
# All 18 datasets (12 2D + 6 3D):
python main_medmnist.py \
    --model swin_tiny \
    --data_sets all \
    --pretrain_epochs 50 \
    --batch_size 128 \
    --lr 1e-3 \
    --exp_name run01

# 12 × 2D only:
python main_medmnist.py --model swin_tiny --data_sets 2d --exp_name 2d_only

# 6 × 3D only:
python main_medmnist.py --model swin_tiny --data_sets 3d --exp_name 3d_only

# Hand-pick datasets:
python main_medmnist.py \
    --model swin_tiny \
    --data_set PathMNIST --data_set BloodMNIST --data_set OrganMNIST3D \
    --exp_name custom
"""

import os
import sys
import argparse

import torch

from utils import get_config
from dataloader_medmnist import (
    build_medmnist_datasets,
    MEDMNIST_2D_FLAGS,
    MEDMNIST_3D_FLAGS,
)
from engine_medmnist import omni_engine_medmnist

sys.setrecursionlimit(40000)


# ── Dataset name → flag mapping (YAML key → medmnist flag) ──────────────────
# These keys match datasets_config_medmnist.yaml exactly.
_NAME_TO_FLAG = {
    'PathMNIST':     'pathmnist',
    'ChestMNIST':    'chestmnist',
    'DermaMNIST':    'dermamnist',
    'OCTMNIST':      'octmnist',
    'PneumoniaMNIST':'pneumoniamnist',
    'RetinaMNIST':   'retinamnist',
    'BreastMNIST':   'breastmnist',
    'BloodMNIST':    'bloodmnist',
    'TissueMNIST':   'tissuemnist',
    'OrganAMNIST':   'organamnist',
    'OrganCMNIST':   'organcmnist',
    'OrganSMNIST':   'organsmnist',
    # 3D
    'OrganMNIST3D':  'organmnist3d',
    'NoduleMNIST3D': 'nodulemnist3d',
    'AdrenalMNIST3D':'adrenalmnist3d',
    'FractureMNIST3D':'fracturemnist3d',
    'VesselMNIST3D': 'vesselmnist3d',
    'SynapseMNIST3D':'synapsemnist3d',
}

_ALL_2D_NAMES = ['PathMNIST','ChestMNIST','DermaMNIST','OCTMNIST',
                 'PneumoniaMNIST','RetinaMNIST','BreastMNIST','BloodMNIST',
                 'TissueMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST']

_ALL_3D_NAMES = ['OrganMNIST3D','NoduleMNIST3D','AdrenalMNIST3D',
                 'FractureMNIST3D','VesselMNIST3D','SynapseMNIST3D']

_ALL_18_NAMES = _ALL_2D_NAMES + _ALL_3D_NAMES


# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Ark+ cyclic pretraining on MedMNIST 18 datasets')

    # ── Dataset selection ────────────────────────────────────────────────────
    p.add_argument('--data_sets', default='all',
                   help='"all" | "2d" | "3d" or one/more --data_set flags')
    p.add_argument('--data_set', dest='dataset_names',
                   action='append', default=[],
                   metavar='DATASET',
                   help='Specific dataset(s) to include; repeatable. '
                        'Overrides --data_sets if any are given.')
    p.add_argument('--medmnist_root', default=None,
                   help='Cache dir for medmnist downloads (default ~/.medmnist)')
    p.add_argument('--img_size', type=int, default=28,
                   help='Image resolution for 2D datasets (28|64|128|224)')

    # ── Model ────────────────────────────────────────────────────────────────
    p.add_argument('--model', dest='model_name', default='swin_tiny',
                   choices=['swin_tiny','swin_small','swin_base',
                            'swin_large','conv_base'])
    p.add_argument('--pretrained_weights', default=None)
    p.add_argument('--projector_features', type=int, default=None)
    p.add_argument('--use_mlp', action='store_true', default=False)
    p.add_argument('--reinit_heads', action='store_true', default=False)

    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument('--pretrain_epochs', type=int, default=50)
    p.add_argument('--batch_size',      type=int, default=128)
    p.add_argument('--workers',         type=int, default=8)
    p.add_argument('--device',          default='cuda')
    p.add_argument('--exp_name',        default='exp01')
    p.add_argument('--mode',            default='train', choices=['train','test'])
    p.add_argument('--resume',          action='store_true', default=False)
    p.add_argument('--test_epoch',      type=int, default=5)
    p.add_argument('--val_loss_metric', default='average')

    # ── EMA ──────────────────────────────────────────────────────────────────
    p.add_argument('--ema_mode',          default='epoch',
                   choices=['epoch','iteration'])
    p.add_argument('--momentum_teacher',  type=float, default=0.9)

    # ── Optimiser (timm-compatible) ─────────────────────────────────────────
    p.add_argument('--opt',            default='adamw')
    p.add_argument('--opt_eps',        type=float, default=1e-8)
    p.add_argument('--opt_betas',      type=float, nargs='+', default=None)
    p.add_argument('--clip_grad',      type=float, default=None)
    p.add_argument('--momentum',       type=float, default=0.9)
    p.add_argument('--weight_decay',   type=float, default=0.05)

    # ── LR schedule (timm-compatible) ───────────────────────────────────────
    p.add_argument('--sched',          default='cosine')
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--min_lr',         type=float, default=1e-5)
    p.add_argument('--warmup_lr',      type=float, default=1e-6)
    p.add_argument('--warmup_epochs',  type=int,   default=5)
    p.add_argument('--cooldown_epochs',type=int,   default=10)
    p.add_argument('--decay_epochs',   type=float, default=30)
    p.add_argument('--decay_rate',     type=float, default=0.5)
    p.add_argument('--patience_epochs',type=int,   default=10)
    p.add_argument('--lr_noise',       type=float, nargs='+', default=None)
    p.add_argument('--lr_noise_pct',   type=float, default=0.67)
    p.add_argument('--lr_noise_std',   type=float, default=1.0)

    # timm create_optimizer needs these attribute names (with hyphens replaced)
    args = p.parse_args()

    # timm compat aliases
    args.opt_eps   = args.opt_eps
    args.clip_grad = args.clip_grad

    # Keep crop_size alias so models_medmnist can read it
    args.crop_size = args.img_size

    return args


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    print(args)

    # ── Resolve dataset list ──────────────────────────────────────────────────
    if args.dataset_names:
        # Explicit --data_set flags take priority
        dataset_list = args.dataset_names
    elif args.data_sets == 'all':
        dataset_list = _ALL_18_NAMES
    elif args.data_sets == '2d':
        dataset_list = _ALL_2D_NAMES
    elif args.data_sets == '3d':
        dataset_list = _ALL_3D_NAMES
    else:
        raise ValueError(f"Unknown --data_sets value: {args.data_sets!r}. "
                         "Use 'all', '2d', '3d', or explicit --data_set flags.")

    print(f"Training on {len(dataset_list)} datasets: {dataset_list}")

    # ── Load config ────────────────────────────────────────────────────────────
    config_path    = os.path.join(os.path.dirname(__file__),
                                  'datasets_config_medmnist.yaml')
    datasets_config = get_config(config_path)

    for ds in dataset_list:
        assert ds in datasets_config, \
            f"Dataset '{ds}' not found in config. " \
            f"Available: {list(datasets_config.keys())}"

    # ── Build datasets ────────────────────────────────────────────────────────
    dataset_train_list, dataset_val_list, dataset_test_list = [], [], []

    for ds_name in dataset_list:
        flag = _NAME_TO_FLAG[ds_name]
        tr, vl, te = build_medmnist_datasets(
            flag=flag,
            split=None,          # build_medmnist_datasets handles all splits
            size=args.img_size,
            download=True,
            root=args.medmnist_root,
        )
        dataset_train_list.append(tr)
        dataset_val_list.append(vl)
        dataset_test_list.append(te)

    # ── Output dirs ────────────────────────────────────────────────────────────
    model_path  = os.path.join('./Models',  f'{args.model_name}_{args.exp_name}')
    output_path = os.path.join('./Outputs', f'{args.model_name}_{args.exp_name}')

    # ── Launch engine ──────────────────────────────────────────────────────────
    omni_engine_medmnist(
        args,
        model_path, output_path,
        dataset_list,
        datasets_config,
        dataset_train_list,
        dataset_val_list,
        dataset_test_list,
    )


if __name__ == '__main__':
    main()
