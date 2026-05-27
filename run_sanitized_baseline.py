"""
Train a PACA-guided sanitized model on the STANDARD train/test split.

This produces a clean sanitized checkpoint for C2c (re-attribution) and C2e (compression),
avoiding the confound of IP-stratified splits.

Usage:
  python -m paca.run_sanitized_baseline --dataset USTC --seed 42

Saves:
  output/baselines_sanitized/{dataset}/best_checkpoint.pth
"""

import argparse
import json
import os
import sys
import time
import datetime
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy

sys.path.insert(0, str(Path(__file__).parent.parent / "YaTC"))

import models_YaTC
import util.lr_decay as lrd
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine import train_one_epoch, evaluate

from paca.protocol_parser import MFRProtocolMapper, ATTRIBUTION_FIELDS

DATASET_CONFIGS = {
    "USTC":       {"data_path": "YaTC_datasets/USTC-TFC2016_MFR",  "nb_classes": 20},
    "ISCX-VPN":   {"data_path": "YaTC_datasets/ISCXVPN2016_MFR",   "nb_classes": 7},
    "ISCX-Tor":   {"data_path": "YaTC_datasets/ISCXTor2016_MFR",   "nb_classes": 8},
    "CSTNET":     {"data_path": "YaTC_datasets/CSTNET-TLS1.3_MFR", "nb_classes": 119},
    "CICIoT2022": {"data_path": "YaTC_datasets/CICIoT2022_MFR",    "nb_classes": 10},
}


class SanitizedImageFolder(torch.utils.data.Dataset):
    """ImageFolder-like dataset that randomizes specified byte indices during training."""

    def __init__(self, root_dir, sanitize_indices, transform=None, is_train=True):
        self.transform = transform
        self.sanitize_indices = sanitize_indices
        self.is_train = is_train
        self.samples = []
        self.labels = []

        classes = sorted([c for c in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, c))])
        for ci, cn in enumerate(classes):
            cd = os.path.join(root_dir, cn)
            for f in sorted(os.listdir(cd)):
                if f.endswith('.png'):
                    self.samples.append(os.path.join(cd, f))
                    self.labels.append(ci)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('L')
        arr = np.array(img, dtype=np.uint8).flatten()
        if self.sanitize_indices and self.is_train:
            arr[self.sanitize_indices] = np.random.randint(
                0, 256, size=len(self.sanitize_indices), dtype=np.uint8)
        img = Image.fromarray(arr.reshape(40, 40), mode='L')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_sanitize_indices(attr_path, theta, mapper):
    with open(attr_path) as f:
        attr = json.load(f)
    fa = attr["field_attribution"]
    all_fields = {f.name: f for f in ATTRIBUTION_FIELDS}
    scores = {fn: fa[fn]["A_conservative"] for fn in fa if fn in all_fields}
    threshold = np.percentile(list(scores.values()), theta)
    indices, fields = [], []
    for fn, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if sc >= threshold and fn in all_fields:
            fields.append(fn)
            indices.extend(mapper.get_field_indices_all_packets(all_fields[fn]))
    return sorted(set(indices)), fields


def get_args():
    p = argparse.ArgumentParser("Train sanitized baseline on standard split")
    p.add_argument("--dataset",    required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--theta",      default=75, type=int)
    p.add_argument("--epochs",     default=50, type=int)
    p.add_argument("--patience",   default=15, type=int)
    p.add_argument("--min_epochs", default=25, type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--blr",        default=2e-3, type=float)
    p.add_argument("--warmup_epochs", default=20, type=int)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--seed",       default=42, type=int)
    p.add_argument("--project_root", default=None)
    return p.parse_args()


def main():
    args = get_args()
    cfg  = DATASET_CONFIGS[args.dataset]
    root = Path(args.project_root) if args.project_root else Path(__file__).parent.parent

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    device = torch.device(args.device)

    data_path = str(root / cfg["data_path"])
    output_dir = root / "output" / "baselines_sanitized" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get PACA sanitization indices
    mapper = MFRProtocolMapper()
    attr_path = str(root / "output" / "attribution" / "field" / args.dataset / "mode_B" / "results.json")
    san_indices, san_fields = get_sanitize_indices(attr_path, args.theta, mapper)
    print(f"=== Sanitized Baseline: {args.dataset} ===")
    print(f"Sanitize fields ({len(san_fields)}): {san_fields}")
    print(f"Sanitize bytes: {len(san_indices)}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_ds = SanitizedImageFolder(os.path.join(data_path, 'train'), san_indices, transform, is_train=True)
    test_ds  = SanitizedImageFolder(os.path.join(data_path, 'test'),  [],           transform, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    # Build model from pretrained
    pretrained = str(root / "YaTC" / "YaTC_pretrained_model.pth")
    model = models_YaTC.TraFormer_YaTC(num_classes=cfg["nb_classes"], drop_path_rate=0.1)
    ckpt = torch.load(pretrained, map_location='cpu', weights_only=False)
    state = model.state_dict()
    ckpt_model = ckpt['model']
    for k in ['head.weight', 'head.bias']:
        if k in ckpt_model and ckpt_model[k].shape != state[k].shape:
            del ckpt_model[k]
    interpolate_pos_embed(model, ckpt_model)
    model.load_state_dict(ckpt_model, strict=False)
    trunc_normal_(model.head.weight, std=2e-5)
    model.to(device)

    lr = args.blr * args.batch_size / 256
    param_groups = lrd.param_groups_lrd(model, 0.05,
                                         no_weight_decay_list=model.no_weight_decay(),
                                         layer_decay=0.75)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    loss_scaler = NativeScaler()
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    import argparse as _ap
    engine_args = _ap.Namespace(
        accum_iter=1, clip_grad=None, lr=lr, blr=args.blr,
        min_lr=1e-6, warmup_epochs=args.warmup_epochs, epochs=args.epochs,
        output_dir=str(output_dir), distributed=False)

    best_f1, best_epoch, no_improve = 0.0, 0, 0

    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, train_loader, optimizer, device,
                        epoch, loss_scaler, max_norm=None, mixup_fn=None,
                        log_writer=None, args=engine_args)
        stats = evaluate(test_loader, model, device)
        f1 = stats['macro_f1']
        acc = stats['acc1']

        if f1 > best_f1:
            best_f1, best_epoch, no_improve = f1, epoch, 0
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch, 'acc': acc, 'f1': f1,
                'sanitize_fields': san_fields,
                'sanitize_bytes': len(san_indices),
                'theta': args.theta,
            }, str(output_dir / "best_checkpoint.pth"))
            print(f"  => New best: acc={acc:.4f}, F1={f1:.4f}")
        else:
            no_improve += 1

        if epoch >= args.min_epochs and no_improve >= args.patience:
            print(f"Early stop at epoch {epoch}, best={best_epoch}")
            break

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: acc={acc:.4f}, F1={f1:.4f} (best={best_f1:.4f})")

    print(f"\nBest: epoch={best_epoch}, F1={best_f1:.4f}")

    # Save results
    results = {
        "dataset": args.dataset,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "sanitize_fields": san_fields,
        "sanitize_bytes": len(san_indices),
        "theta": args.theta,
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {output_dir}")


if __name__ == "__main__":
    main()