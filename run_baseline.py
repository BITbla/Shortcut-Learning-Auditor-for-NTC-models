"""
Block 0: Baseline Model Fine-tuning for all datasets.

Usage:
  python -m paca.run_baseline --dataset USTC --seed 42
  python -m paca.run_baseline --dataset ISCX-VPN --seed 42
  python -m paca.run_baseline --dataset ISCX-Tor --seed 42

Saves:
  - Best checkpoint: output/baselines/{dataset}/best_checkpoint.pth
  - Results JSON:    output/baselines/{dataset}/results.json
"""

import argparse
import json
import os
import sys
import time
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import timm
try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# Add YaTC to path
YATC_DIR = Path(__file__).parent.parent / "YaTC"
sys.path.insert(0, str(YATC_DIR))

import models_YaTC
import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine import train_one_epoch, evaluate

from sklearn.metrics import classification_report

# Dataset configurations
DATASET_CONFIGS = {
    "USTC": {
        "data_path": "YaTC_datasets/USTC-TFC2016_MFR",
        "nb_classes": 20,
        "target_acc": 0.95,
    },
    "ISCX-VPN": {
        "data_path": "YaTC_datasets/ISCXVPN2016_MFR",
        "nb_classes": 7,
        "target_acc": 0.90,
    },
    "ISCX-Tor": {
        "data_path": "YaTC_datasets/ISCXTor2016_MFR",
        "nb_classes": 8,
        "target_acc": 0.80,
    },
    "CSTNET": {
        "data_path": "YaTC_datasets/CSTNET-TLS1.3_MFR",
        "nb_classes": 119,
        "target_acc": 0.70,
    },
    "CICIoT2022": {
        "data_path": "YaTC_datasets/CICIoT2022_MFR",
        "nb_classes": 10,
        "target_acc": 0.80,
    },
    "CSTNET_san": {
        "data_path": "YaTC_datasets/CSTNET-TLS1.3_MFR_san",
        "nb_classes": 119,
        "target_acc": 0.70,
    },
    "USTC_san": {
        "data_path": "YaTC_datasets/USTC-TFC2016_MFR_san",
        "nb_classes": 20,
        "target_acc": 0.90,
    },
    "CICIoT2022_san": {
        "data_path": "YaTC_datasets/CICIoT2022_MFR_san",
        "nb_classes": 10,
        "target_acc": 0.70,
    },
}


def get_args():
    parser = argparse.ArgumentParser("PACA Baseline Fine-tuning")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset name")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--blr", default=2e-3, type=float, help="Base learning rate")
    parser.add_argument("--warmup_epochs", default=20, type=int)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing")
    parser.add_argument("--layer_decay", default=0.75, type=float)
    parser.add_argument("--drop_path", default=0.1, type=float)
    parser.add_argument("--pretrained", default=None, type=str,
                        help="Path to pretrained model (default: auto-detect)")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--project_root", default=None, type=str,
                        help="Project root directory (default: auto-detect)")
    parser.add_argument("--patience", default=15, type=int,
                        help="Early stopping: stop if F1 does not improve for this many epochs")
    parser.add_argument("--min_epochs", default=25, type=int,
                        help="Minimum epochs before early stopping can trigger")
    return parser.parse_args()


def build_dataset(data_path, is_train):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    root = os.path.join(data_path, 'train' if is_train else 'test')
    return datasets.ImageFolder(root, transform=transform)


def main():
    args = get_args()
    cfg = DATASET_CONFIGS[args.dataset]

    # Resolve project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        project_root = Path(__file__).parent.parent

    data_path = project_root / cfg["data_path"]
    if not data_path.exists():
        print(f"ERROR: Dataset path not found: {data_path}")
        sys.exit(1)

    # Pretrained model path
    if args.pretrained:
        pretrained_path = args.pretrained
    else:
        pretrained_path = str(project_root / "YaTC" / "YaTC_pretrained_model.pth")

    # Output directory
    output_dir = project_root / "output" / "baselines" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Baseline Fine-tuning: {args.dataset} ===")
    print(f"Data path: {data_path}")
    print(f"Pretrained: {pretrained_path}")
    print(f"Output: {output_dir}")
    print(f"Classes: {cfg['nb_classes']}, Target acc: {cfg['target_acc']}")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Build datasets
    dataset_train = build_dataset(str(data_path), is_train=True)
    dataset_val = build_dataset(str(data_path), is_train=False)
    class_names = dataset_val.classes
    print(f"Train: {len(dataset_train)} samples, Test: {len(dataset_val)} samples")
    print(f"Classes: {class_names}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )

    # Build model
    model = models_YaTC.TraFormer_YaTC(
        num_classes=cfg["nb_classes"], drop_path_rate=args.drop_path)

    # Load pretrained weights
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Loaded pretrained: {msg}")
    trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_parameters / 1e6:.2f}M")

    # Optimizer with layer-wise lr decay
    eff_batch_size = args.batch_size
    lr = args.blr * eff_batch_size / 256
    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
                                         no_weight_decay_list=model.no_weight_decay(),
                                         layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    loss_scaler = NativeScaler()

    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    log_writer = SummaryWriter(log_dir=str(log_dir))

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    max_accuracy = 0.0
    max_f1 = 0.0
    best_epoch = 0
    no_improve_count = 0

    # Create a minimal args namespace for engine compatibility
    engine_args = argparse.Namespace(
        accum_iter=1, clip_grad=None, lr=lr, blr=args.blr,
        min_lr=1e-6, warmup_epochs=args.warmup_epochs, epochs=args.epochs,
        output_dir=str(output_dir), distributed=False,
    )

    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None, mixup_fn=None,
            log_writer=log_writer, args=engine_args,
        )

        test_stats = evaluate(data_loader_val, model, device)
        acc = test_stats['acc1']
        f1 = test_stats['macro_f1']

        print(f"Epoch {epoch}: acc={acc:.4f}, F1={f1:.4f}")

        if f1 > max_f1:
            max_f1 = f1
            max_accuracy = acc
            best_epoch = epoch
            no_improve_count = 0
            # Save best checkpoint
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'acc': acc,
                'f1': f1,
                'args': vars(args),
            }, str(output_dir / "best_checkpoint.pth"))
            print(f"  => New best: acc={acc:.4f}, F1={f1:.4f}")
        else:
            no_improve_count += 1

        log_writer.add_scalar('test/acc', acc, epoch)
        log_writer.add_scalar('test/f1', f1, epoch)
        log_writer.add_scalar('train/loss', train_stats.get('loss', 0), epoch)

        # Early stopping
        if epoch >= args.min_epochs and no_improve_count >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no F1 improvement for {args.patience} epochs, best was epoch {best_epoch})")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {datetime.timedelta(seconds=int(total_time))}")
    print(f"Best: epoch={best_epoch}, acc={max_accuracy:.4f}, F1={max_f1:.4f}")

    # Final evaluation with per-class report
    model_best = models_YaTC.TraFormer_YaTC(
        num_classes=cfg["nb_classes"], drop_path_rate=args.drop_path)
    ckpt = torch.load(str(output_dir / "best_checkpoint.pth"), map_location='cpu', weights_only=False)
    model_best.load_state_dict(ckpt['model'])
    model_best.to(device)
    model_best.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in data_loader_val:
            images = images.to(device)
            outputs = model_best(images)
            _, preds = outputs.topk(1, 1, True, True)
            all_preds.extend(preds.squeeze().cpu().numpy())
            all_targets.extend(targets.numpy())

    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(all_targets, all_preds, target_names=class_names)
    print(f"\n{report_str}")

    # Save results
    results = {
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "accuracy": float(max_accuracy),
        "macro_f1": float(max_f1),
        "target_acc": cfg["target_acc"],
        "target_met": max_accuracy >= cfg["target_acc"],
        "per_class": {cn: {"precision": report[cn]["precision"],
                           "recall": report[cn]["recall"],
                           "f1": report[cn]["f1-score"]}
                      for cn in class_names},
        "class_names": class_names,
        "training_time_seconds": total_time,
    }

    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Check target
    if max_accuracy >= cfg["target_acc"]:
        print(f"SUCCESS: Target accuracy {cfg['target_acc']} met ({max_accuracy:.4f})")
    else:
        print(f"WARNING: Target accuracy {cfg['target_acc']} NOT met ({max_accuracy:.4f})")


if __name__ == "__main__":
    main()
