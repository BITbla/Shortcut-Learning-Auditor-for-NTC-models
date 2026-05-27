"""
Block 2: Fine-Grained Field-Level Attribution (PACA Core).

Usage:
  python -m paca.run_field_attribution --dataset USTC --checkpoint output/baselines/USTC/best_checkpoint.pth
  python -m paca.run_field_attribution --dataset USTC --checkpoint ... --mode B  # region isolation

Saves:
  - Results JSON: output/attribution/field/{dataset}/mode_{A|B}/results.json
  - Raw scores:   output/attribution/field/{dataset}/mode_{A|B}/raw_scores.npz
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "YaTC"))

from paca.utils import load_yatc_model, load_mfr_dataset_raw
from paca.attribution import PACAttributor
from paca.protocol_parser import (
    get_all_fields, ATTRIBUTION_FIELDS, PAYLOAD_FIELD,
    KNOWN_SHORTCUT_FIELDS, KNOWN_SEMANTIC_FIELDS, AMBIGUOUS_FIELDS,
)

DATASET_CONFIGS = {
    "USTC": {"data_path": "YaTC_datasets/USTC-TFC2016_MFR", "nb_classes": 20},
    "ISCX-VPN": {"data_path": "YaTC_datasets/ISCXVPN2016_MFR", "nb_classes": 7},
    "ISCX-Tor": {"data_path": "YaTC_datasets/ISCXTor2016_MFR", "nb_classes": 8},
    "CSTNET": {"data_path": "YaTC_datasets/CSTNET-TLS1.3_MFR", "nb_classes": 119},
    "CICIoT2022": {"data_path": "YaTC_datasets/CICIoT2022_MFR", "nb_classes": 10},
    "CSTNET_san": {"data_path": "YaTC_datasets/CSTNET-TLS1.3_MFR_san", "nb_classes": 119},
    "USTC_san": {"data_path": "YaTC_datasets/USTC-TFC2016_MFR_san", "nb_classes": 20},
    "CICIoT2022_san": {"data_path": "YaTC_datasets/CICIoT2022_MFR_san", "nb_classes": 10},
}


def get_args():
    parser = argparse.ArgumentParser("PACA Field-Level Attribution")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--mode", default="A", choices=["A", "B"],
                        help="A=global, B=region-isolated")
    parser.add_argument("--R", default=50, type=int)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--project_root", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def compute_expert_correlation(field_scores: dict) -> dict:
    """Compute rank correlation with expert labels.

    Expert ranking: known_shortcut fields should have HIGHER attribution
    than known_semantic fields. We assign numeric labels:
      known_shortcut -> 2, ambiguous -> 1, known_semantic -> 0
    and compute Spearman correlation with A_conservative.
    """
    field_names = []
    attr_values = []
    expert_values = []

    for fname, score in field_scores.items():
        if fname in KNOWN_SHORTCUT_FIELDS:
            expert_val = 2
        elif fname in AMBIGUOUS_FIELDS:
            expert_val = 1
        elif fname in KNOWN_SEMANTIC_FIELDS:
            expert_val = 0
        else:
            continue
        field_names.append(fname)
        attr_values.append(score)
        expert_values.append(expert_val)

    if len(field_names) < 3:
        return {"spearman_r": float('nan'), "spearman_p": float('nan'), "n_fields": len(field_names)}

    r, p = stats.spearmanr(attr_values, expert_values)
    return {"spearman_r": float(r), "spearman_p": float(p), "n_fields": len(field_names)}


def compute_auc_roc(field_scores: dict) -> float:
    """Compute AUC-ROC for shortcut vs semantic classification."""
    from sklearn.metrics import roc_auc_score

    scores = []
    binary_labels = []  # 1 = shortcut, 0 = semantic

    for fname, score in field_scores.items():
        if fname in KNOWN_SHORTCUT_FIELDS:
            scores.append(score)
            binary_labels.append(1)
        elif fname in KNOWN_SEMANTIC_FIELDS:
            scores.append(score)
            binary_labels.append(0)

    if len(set(binary_labels)) < 2:
        return float('nan')

    return float(roc_auc_score(binary_labels, scores))


def main():
    args = get_args()
    cfg = DATASET_CONFIGS[args.dataset]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent.parent
    data_path = str(project_root / cfg["data_path"])

    output_dir = project_root / "output" / "attribution" / "field" / args.dataset / f"mode_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Field-Level Attribution: {args.dataset}, Mode {args.mode} ===")

    # Load model and data
    model = load_yatc_model(args.checkpoint, cfg["nb_classes"], args.device)

    print("Loading dataset...")
    test_samples, test_labels, class_names = load_mfr_dataset_raw(data_path, split='test')
    train_samples, train_labels, _ = load_mfr_dataset_raw(data_path, split='train')
    print(f"Test: {len(test_samples)}, Train: {len(train_samples)}")

    if args.max_samples and args.max_samples < len(test_samples):
        indices = np.random.choice(len(test_samples), args.max_samples, replace=False)
        test_samples = test_samples[indices]
        test_labels = test_labels[indices]
        print(f"Subsampled to {len(test_samples)} test samples")

    attributor = PACAttributor(
        model=model, device=args.device, num_classes=cfg["nb_classes"],
        dataset_samples=train_samples, dataset_labels=train_labels,
    )

    # Fields to attribute
    fields = ATTRIBUTION_FIELDS + [PAYLOAD_FIELD]
    field_names = [f.name for f in fields]
    strategies = ['zero', 'one', 'rand', 'cross']

    # Storage: field_name -> strategy -> list of per-sample scores
    all_scores = {fn: {s: [] for s in strategies} for fn in field_names}

    start_time = time.time()
    for i in tqdm(range(len(test_samples)), desc="Field attribution"):
        mfr = test_samples[i]
        label = int(test_labels[i])

        for field in fields:
            # Determine region isolation for Mode B
            if args.mode == "B":
                if field.header_offset < 80:
                    isolation = 'isolate_payload'
                else:
                    isolation = 'isolate_header'
            else:
                isolation = None

            attr = attributor.compute_field_attribution(
                mfr, label, i, field,
                strategies=strategies, R=args.R,
                region_isolation=isolation,
                batch_size=args.batch_size,
            )
            for s in strategies:
                all_scores[field.name][s].append(attr[s])

    elapsed = time.time() - start_time
    print(f"Attribution complete in {elapsed:.1f}s ({elapsed/len(test_samples):.2f}s/sample)")

    # Aggregate: mean, std, 95% CI, A_conservative
    aggregated = {}
    for fn in field_names:
        field_agg = {}
        strategy_means = {}
        for s in strategies:
            arr = np.array(all_scores[fn][s])
            n = len(arr)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
            field_agg[s] = {
                "mean": mean, "std": std,
                "ci95_low": mean - ci95, "ci95_high": mean + ci95,
            }
            strategy_means[s] = mean

        # A_conservative = min across strategies
        a_conservative = min(strategy_means.values())
        field_agg["A_conservative"] = a_conservative
        aggregated[fn] = field_agg

    # Rank by A_conservative
    ranked = sorted(aggregated.items(), key=lambda x: x[1]["A_conservative"], reverse=True)
    print("\n=== Field Attribution Ranking (A_conservative) ===")
    for rank, (fn, agg) in enumerate(ranked, 1):
        label_tag = ""
        if fn in KNOWN_SHORTCUT_FIELDS:
            label_tag = " [SHORTCUT]"
        elif fn in KNOWN_SEMANTIC_FIELDS:
            label_tag = " [SEMANTIC]"
        elif fn in AMBIGUOUS_FIELDS:
            label_tag = " [AMBIGUOUS]"
        print(f"  {rank:2d}. {fn:25s} A_cons={agg['A_conservative']:.4f}{label_tag}")

    # Expert correlation
    a_cons_dict = {fn: agg["A_conservative"] for fn, agg in aggregated.items()}
    corr = compute_expert_correlation(a_cons_dict)
    print(f"\nSpearman correlation with expert labels: r={corr['spearman_r']:.4f}, p={corr['spearman_p']:.4f}")

    # AUC-ROC
    auc = compute_auc_roc(a_cons_dict)
    print(f"AUC-ROC (shortcut vs semantic): {auc:.4f}")

    # Cross-strategy consistency (Kendall's tau between strategy pairs)
    strategy_consistency = {}
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i+1:]:
            vals1 = [aggregated[fn][s1]["mean"] for fn in field_names]
            vals2 = [aggregated[fn][s2]["mean"] for fn in field_names]
            tau, p = stats.kendalltau(vals1, vals2)
            key = f"{s1}_vs_{s2}"
            strategy_consistency[key] = {"tau": float(tau), "p": float(p)}
            print(f"Kendall's tau ({s1} vs {s2}): {tau:.4f} (p={p:.4f})")

    # Shortcut vs semantic group comparison (Mann-Whitney U)
    shortcut_scores = [a_cons_dict[fn] for fn in a_cons_dict if fn in KNOWN_SHORTCUT_FIELDS]
    semantic_scores = [a_cons_dict[fn] for fn in a_cons_dict if fn in KNOWN_SEMANTIC_FIELDS]
    if shortcut_scores and semantic_scores:
        u_stat, u_p = stats.mannwhitneyu(shortcut_scores, semantic_scores, alternative='greater')
        print(f"\nMann-Whitney U (shortcut > semantic): U={u_stat:.1f}, p={u_p:.4f}")
        group_test = {"U": float(u_stat), "p": float(u_p),
                      "shortcut_mean": float(np.mean(shortcut_scores)),
                      "semantic_mean": float(np.mean(semantic_scores))}
    else:
        group_test = {}

    # Save results
    results = {
        "dataset": args.dataset,
        "mode": args.mode,
        "n_samples": len(test_samples),
        "R": args.R,
        "elapsed_seconds": elapsed,
        "field_attribution": aggregated,
        "ranking": [fn for fn, _ in ranked],
        "expert_correlation": corr,
        "auc_roc": auc,
        "strategy_consistency": strategy_consistency,
        "group_test": group_test,
    }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save raw per-sample scores
    raw_data = {}
    for fn in field_names:
        for s in strategies:
            raw_data[f"{fn}_{s}"] = np.array(all_scores[fn][s])
    raw_data["labels"] = np.array(test_labels[:len(test_samples)])
    np.savez(output_dir / "raw_scores.npz", **raw_data)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
