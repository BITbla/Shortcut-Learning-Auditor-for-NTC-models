"""
Block 7 (C2d): Attribution Ranking Validation — Field Removal Curve.

Progressively zero out fields in order of PACA A_conservative ranking,
and measure F1 at each step. Compare PACA-guided (low→high), Random,
and Reverse-PACA (high→low) orderings.

Usage:
  python -m paca.run_field_removal --dataset USTC --checkpoint output/baselines/USTC/best_checkpoint.pth --attribution_results output/attribution/field/USTC/mode_B/results.json

Saves:
  - output/field_removal/{dataset}/results.json
  - output/field_removal/{dataset}/curve.csv
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent.parent / "YaTC"))

import models_YaTC
from paca.utils import load_yatc_model
from paca.protocol_parser import (
    MFRProtocolMapper, ATTRIBUTION_FIELDS, PAYLOAD_FIELD,
    KNOWN_SHORTCUT_FIELDS, KNOWN_SEMANTIC_FIELDS,
)

DATASET_CONFIGS = {
    "USTC": {"data_path": "YaTC_datasets/USTC-TFC2016_MFR", "nb_classes": 20},
    "ISCX-VPN": {"data_path": "YaTC_datasets/ISCXVPN2016_MFR", "nb_classes": 7},
    "ISCX-Tor": {"data_path": "YaTC_datasets/ISCXTor2016_MFR", "nb_classes": 8},
    "CSTNET": {"data_path": "YaTC_datasets/CSTNET-TLS1.3_MFR", "nb_classes": 119},
    "CICIoT2022": {"data_path": "YaTC_datasets/CICIoT2022_MFR", "nb_classes": 10},
    "USTC_san": {"data_path": "YaTC_datasets/USTC-TFC2016_MFR_san", "nb_classes": 20},
    "CSTNET_san": {"data_path": "YaTC_datasets/CSTNET-TLS1.3_MFR_san", "nb_classes": 119},
    "CICIoT2022_san": {"data_path": "YaTC_datasets/CICIoT2022_MFR_san", "nb_classes": 10},
}


def get_args():
    parser = argparse.ArgumentParser("PACA Field Removal Curve")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--attribution_results", required=True, type=str,
                        help="Path to field attribution results.json (Mode B recommended)")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--project_root", default=None, type=str)
    parser.add_argument("--random_seeds", default=5, type=int,
                        help="Number of random orderings to average")
    return parser.parse_args()


class FieldMaskedDataset(torch.utils.data.Dataset):
    """Dataset that zeros out specified field byte positions."""

    def __init__(self, base_dataset_path: str, split: str,
                 zero_indices: list, transform=None):
        self.transform = transform
        self.zero_indices = zero_indices

        split_path = os.path.join(base_dataset_path, split)
        classes = sorted([c for c in os.listdir(split_path)
                          if os.path.isdir(os.path.join(split_path, c))])
        self.samples = []
        self.labels = []
        for ci, cn in enumerate(classes):
            cd = os.path.join(split_path, cn)
            for f in sorted(os.listdir(cd)):
                if f.endswith('.png'):
                    img = Image.open(os.path.join(cd, f)).convert('L')
                    self.samples.append(np.array(img, dtype=np.uint8))
                    self.labels.append(ci)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr = self.samples[idx].copy().flatten()
        if self.zero_indices:
            arr[self.zero_indices] = 0
        img = Image.fromarray(arr.reshape(40, 40), mode='L')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def evaluate_with_mask(model, data_path, zero_indices, device, batch_size, num_workers):
    """Evaluate model with specified byte positions zeroed out."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = FieldMaskedDataset(data_path, 'test', zero_indices, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.topk(1, 1, True, True)
            all_preds.extend(preds.squeeze().cpu().numpy())
            all_targets.extend(targets.numpy())

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return float(acc), float(f1)


def main():
    args = get_args()
    cfg = DATASET_CONFIGS[args.dataset]
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent.parent
    data_path = str(project_root / cfg["data_path"])

    output_dir = project_root / "output" / "field_removal" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Field Removal Curve: {args.dataset} ===")

    # Load attribution results
    with open(args.attribution_results) as f:
        attr_results = json.load(f)

    field_attr = attr_results["field_attribution"]
    mapper = MFRProtocolMapper()

    # Build field list with A_conservative scores (header fields only, exclude payload)
    header_fields = []
    all_fields_map = {f.name: f for f in ATTRIBUTION_FIELDS}
    for fn, scores in field_attr.items():
        if fn in all_fields_map and fn != "Encrypted_Payload":
            header_fields.append((fn, scores["A_conservative"], all_fields_map[fn]))

    # Sort by A_conservative
    header_fields_sorted = sorted(header_fields, key=lambda x: x[1])

    print(f"Header fields ({len(header_fields_sorted)}):")
    for fn, ac, _ in header_fields_sorted:
        tag = " [S]" if fn in KNOWN_SHORTCUT_FIELDS else (" [M]" if fn in KNOWN_SEMANTIC_FIELDS else "")
        print(f"  {fn:25s} A_cons={ac:.4f}{tag}")

    # Load model
    model = load_yatc_model(args.checkpoint, cfg["nb_classes"], args.device)

    # Baseline: no masking
    base_acc, base_f1 = evaluate_with_mask(model, data_path, [], args.device, args.batch_size, args.num_workers)
    print(f"\nBaseline: acc={base_acc:.4f}, F1={base_f1:.4f}")

    # === PACA-guided: low→high (remove least important first) ===
    print("\n--- PACA-guided (low→high) ---")
    paca_curve = [{"n_removed": 0, "acc": base_acc, "f1": base_f1, "fields_removed": []}]
    cumulative_indices = []
    for i, (fn, ac, field) in enumerate(header_fields_sorted):
        indices = mapper.get_field_indices_all_packets(field)
        cumulative_indices.extend(indices)
        acc, f1 = evaluate_with_mask(model, data_path, cumulative_indices, args.device, args.batch_size, args.num_workers)
        paca_curve.append({"n_removed": i+1, "acc": acc, "f1": f1, "fields_removed": [x[0] for x in header_fields_sorted[:i+1]]})
        print(f"  Remove {i+1}/{len(header_fields_sorted)} ({fn}): acc={acc:.4f}, F1={f1:.4f}, delta={f1-base_f1:+.4f}")

    # === Reverse-PACA: high→low (remove most important first) ===
    print("\n--- Reverse-PACA (high→low) ---")
    reverse_sorted = list(reversed(header_fields_sorted))
    reverse_curve = [{"n_removed": 0, "acc": base_acc, "f1": base_f1, "fields_removed": []}]
    cumulative_indices = []
    for i, (fn, ac, field) in enumerate(reverse_sorted):
        indices = mapper.get_field_indices_all_packets(field)
        cumulative_indices.extend(indices)
        acc, f1 = evaluate_with_mask(model, data_path, cumulative_indices, args.device, args.batch_size, args.num_workers)
        reverse_curve.append({"n_removed": i+1, "acc": acc, "f1": f1, "fields_removed": [x[0] for x in reverse_sorted[:i+1]]})
        print(f"  Remove {i+1}/{len(reverse_sorted)} ({fn}): acc={acc:.4f}, F1={f1:.4f}, delta={f1-base_f1:+.4f}")

    # === Random orderings ===
    print(f"\n--- Random ({args.random_seeds} seeds) ---")
    random_curves = []
    for seed in range(args.random_seeds):
        np.random.seed(seed)
        shuffled = list(header_fields_sorted)
        np.random.shuffle(shuffled)
        curve = [{"n_removed": 0, "f1": base_f1}]
        cumulative_indices = []
        for i, (fn, ac, field) in enumerate(shuffled):
            indices = mapper.get_field_indices_all_packets(field)
            cumulative_indices.extend(indices)
            acc, f1 = evaluate_with_mask(model, data_path, cumulative_indices, args.device, args.batch_size, args.num_workers)
            curve.append({"n_removed": i+1, "f1": f1})
        random_curves.append(curve)
        print(f"  Seed {seed}: final F1={curve[-1]['f1']:.4f}")

    # Average random curves
    random_avg = []
    for step in range(len(header_fields_sorted) + 1):
        f1s = [rc[step]["f1"] for rc in random_curves]
        random_avg.append({"n_removed": step, "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))})

    # Save results
    results = {
        "dataset": args.dataset,
        "baseline_acc": base_acc,
        "baseline_f1": base_f1,
        "n_header_fields": len(header_fields_sorted),
        "field_order_paca": [x[0] for x in header_fields_sorted],
        "field_order_reverse": [x[0] for x in reverse_sorted],
        "paca_curve": paca_curve,
        "reverse_curve": reverse_curve,
        "random_avg": random_avg,
        "random_seeds": args.random_seeds,
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # CSV for easy plotting
    with open(output_dir / "curve.csv", 'w') as f:
        f.write("n_removed,paca_f1,reverse_f1,random_f1_mean,random_f1_std\n")
        for i in range(len(header_fields_sorted) + 1):
            pf = paca_curve[i]["f1"]
            rf = reverse_curve[i]["f1"]
            rm = random_avg[i]["f1_mean"]
            rs = random_avg[i]["f1_std"]
            f.write(f"{i},{pf:.6f},{rf:.6f},{rm:.6f},{rs:.6f}\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
