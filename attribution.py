"""
PACA Attribution Estimator.

Implements the four masking strategies (T_zero, T_one, T_rand, T_cross)
and computes field-level attribution scores using margin score as target.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from .protocol_parser import (
    MFRProtocolMapper, ProtocolField, get_all_fields,
    ATTRIBUTION_FIELDS, PAYLOAD_FIELD, IP_TCP_FIELDS,
)


class MaskingStrategy:
    """Protocol-conditioned counterfactual replacement operators."""

    def __init__(self, dataset_samples: Optional[np.ndarray] = None,
                 dataset_labels: Optional[np.ndarray] = None):
        """
        Args:
            dataset_samples: (N, 1600) uint8 array of all samples (for T_cross).
            dataset_labels: (N,) int array of labels (for T_cross same-class sampling).
        """
        self.mapper = MFRProtocolMapper()
        self.dataset_samples = dataset_samples
        self.dataset_labels = dataset_labels
        if dataset_labels is not None:
            self._build_class_indices()

    def _build_class_indices(self):
        """Build per-class sample index lists for T_cross."""
        self.class_indices = {}
        for idx, label in enumerate(self.dataset_labels):
            label = int(label)
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

    def apply_zero(self, mfr_flat: np.ndarray, byte_indices: List[int]) -> np.ndarray:
        """T_zero: replace with 0x00."""
        out = mfr_flat.copy()
        out[byte_indices] = 0
        return out

    def apply_one(self, mfr_flat: np.ndarray, byte_indices: List[int]) -> np.ndarray:
        """T_one: replace with 0xFF."""
        out = mfr_flat.copy()
        out[byte_indices] = 255
        return out

    def apply_rand(self, mfr_flat: np.ndarray, byte_indices: List[int],
                   field: Optional[ProtocolField] = None) -> np.ndarray:
        """T_rand: protocol-conditioned random replacement."""
        out = mfr_flat.copy()
        if field is None:
            # Generic random
            out[byte_indices] = np.random.randint(0, 256, size=len(byte_indices), dtype=np.uint8)
            return out

        name = field.name
        for pkt in range(5):
            pkt_base = pkt * 320
            pkt_indices = [i for i in byte_indices if pkt_base <= i < pkt_base + 320]
            if not pkt_indices:
                continue
            local_offsets = [i - pkt_base for i in pkt_indices]

            if name in ("IP_Src", "IP_Dst"):
                # Sample from plausible IPv4 address space
                addr = np.random.randint(1, 255, size=4, dtype=np.uint8)
                vals = np.tile(addr, (len(local_offsets) // 4) + 1)[:len(local_offsets)]
            elif name in ("TCP_Src_Port", "TCP_Dst_Port"):
                port = int(np.random.randint(1024, 65536))
                vals = np.array([(port >> 8) & 0xFF, port & 0xFF], dtype=np.uint8)
                vals = np.tile(vals, (len(local_offsets) // 2) + 1)[:len(local_offsets)]
            elif name in ("TCP_Seq", "TCP_Ack"):
                val = int(np.random.randint(0, 2**31)) * 2 + int(np.random.randint(0, 2))
                vals = np.array([(val >> (8*i)) & 0xFF for i in range(3, -1, -1)], dtype=np.uint8)
                vals = np.tile(vals, (len(local_offsets) // 4) + 1)[:len(local_offsets)]
            elif name == "IP_ID":
                val = int(np.random.randint(0, 65536))
                vals = np.array([(val >> 8) & 0xFF, val & 0xFF], dtype=np.uint8)
                vals = np.tile(vals, (len(local_offsets) // 2) + 1)[:len(local_offsets)]
            elif name == "IP_TTL":
                ttl = int(np.random.choice([32, 64, 128, 255]))
                vals = np.full(len(local_offsets), ttl, dtype=np.uint8)
            elif name == "TCP_DataOff_Flags":
                # Keep original (strong protocol constraint on flags)
                continue
            elif name == "TCP_Window":
                val = int(np.random.randint(0, 65536))
                vals = np.array([(val >> 8) & 0xFF, val & 0xFF], dtype=np.uint8)
                vals = np.tile(vals, (len(local_offsets) // 2) + 1)[:len(local_offsets)]
            elif name == "IP_Checksum":
                val = int(np.random.randint(0, 65536))
                vals = np.array([(val >> 8) & 0xFF, val & 0xFF], dtype=np.uint8)
                vals = np.tile(vals, (len(local_offsets) // 2) + 1)[:len(local_offsets)]
            elif name == "IP_Total_Length":
                val = int(np.random.randint(40, 1500))
                vals = np.array([(val >> 8) & 0xFF, val & 0xFF], dtype=np.uint8)
                vals = np.tile(vals, (len(local_offsets) // 2) + 1)[:len(local_offsets)]
            elif name == "TCP_Timestamp":
                # TSval (4B) + TSecr (4B): sample plausible timestamp values
                tsval = int(np.random.randint(0, 2**31))
                tsecr = int(np.random.randint(0, 2**31))
                vals = np.array([(tsval >> (8*i)) & 0xFF for i in range(3, -1, -1)] +
                                [(tsecr >> (8*i)) & 0xFF for i in range(3, -1, -1)],
                                dtype=np.uint8)
                vals = np.tile(vals, (len(local_offsets) // 8) + 1)[:len(local_offsets)]
            elif name == "Encrypted_Payload":
                vals = np.random.randint(0, 256, size=len(local_offsets), dtype=np.uint8)
            else:
                vals = np.random.randint(0, 256, size=len(local_offsets), dtype=np.uint8)

            out[pkt_indices] = vals[:len(pkt_indices)]
        return out

    def apply_cross(self, mfr_flat: np.ndarray, byte_indices: List[int],
                    label: int, exclude_idx: int) -> np.ndarray:
        """T_cross: replace with same-class other sample's corresponding bytes."""
        out = mfr_flat.copy()
        candidates = [i for i in self.class_indices.get(label, []) if i != exclude_idx]
        if not candidates:
            return out
        donor_idx = np.random.choice(candidates)
        donor = self.dataset_samples[donor_idx]
        out[byte_indices] = donor[byte_indices]
        return out


class PACAttributor:
    """PACA Attribution Estimator for YaTC models."""

    def __init__(self, model, device, num_classes: int,
                 dataset_samples: Optional[np.ndarray] = None,
                 dataset_labels: Optional[np.ndarray] = None):
        """
        Args:
            model: Frozen YaTC model (in eval mode).
            device: torch device.
            num_classes: Number of classification classes.
            dataset_samples: (N, 1600) uint8 for T_cross.
            dataset_labels: (N,) int for T_cross.
        """
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.mapper = MFRProtocolMapper()
        self.masker = MaskingStrategy(dataset_samples, dataset_labels)
        self.model.eval()

    def _mfr_to_tensor(self, mfr_flat: np.ndarray) -> torch.Tensor:
        """Convert flat uint8 MFR (1600,) to model input tensor (1, 1, 40, 40)."""
        img = mfr_flat.reshape(40, 40).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # normalize same as build_dataset
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _margin_score(self, mfr_tensor: torch.Tensor, true_label: int) -> float:
        """Compute margin score: logit_y - max_{c!=y} logit_c."""
        logits = self.model(mfr_tensor)  # (1, num_classes)
        logits = logits.squeeze(0).cpu().numpy()
        true_logit = logits[true_label]
        mask = np.ones(len(logits), dtype=bool)
        mask[true_label] = False
        max_other = logits[mask].max()
        return float(true_logit - max_other)

    @torch.no_grad()
    def _margin_score_batch(self, mfr_tensors: torch.Tensor, true_label: int) -> np.ndarray:
        """Batch margin score for multiple perturbed inputs."""
        logits = self.model(mfr_tensors)  # (B, num_classes)
        logits = logits.cpu().numpy()
        true_logits = logits[:, true_label]
        mask = np.ones(logits.shape[1], dtype=bool)
        mask[true_label] = False
        max_others = logits[:, mask].max(axis=1)
        return true_logits - max_others

    def compute_field_attribution(
        self,
        mfr_flat: np.ndarray,
        true_label: int,
        sample_idx: int,
        field: ProtocolField,
        strategies: List[str] = None,
        R: int = 50,
        region_isolation: Optional[str] = None,
        batch_size: int = 50,
    ) -> Dict[str, float]:
        """
        Compute attribution score for a single field on a single sample.

        Args:
            mfr_flat: (1600,) uint8 array.
            true_label: Ground truth label.
            sample_idx: Index in dataset (for T_cross exclusion).
            field: ProtocolField to attribute.
            strategies: List of strategy names. Default: all four.
            R: Number of replacement repetitions.
            region_isolation: None, 'isolate_payload', or 'isolate_header'.
            batch_size: Batch size for model inference.

        Returns:
            Dict[strategy_name, attribution_score].
        """
        if strategies is None:
            strategies = ['zero', 'one', 'rand', 'cross']

        base_mfr = mfr_flat.copy()
        byte_indices = self.mapper.get_field_indices_all_packets(field, data=base_mfr)

        # Apply region isolation if requested
        if region_isolation == 'isolate_payload':
            # Zero out payload when analyzing header fields
            for pkt in range(5):
                payload_idx = self.mapper.get_payload_byte_indices(pkt)
                base_mfr[payload_idx] = 0
        elif region_isolation == 'isolate_header':
            # Zero out header when analyzing payload fields
            for pkt in range(5):
                header_idx = self.mapper.get_header_byte_indices(pkt)
                base_mfr[header_idx] = 0

        # Original margin score (with isolation applied)
        orig_tensor = self._mfr_to_tensor(base_mfr)
        s_orig = self._margin_score(orig_tensor, true_label)

        results = {}
        for strategy in strategies:
            # Generate R perturbed versions
            perturbed_list = []
            for _ in range(R):
                if strategy == 'zero':
                    p = self.masker.apply_zero(base_mfr, byte_indices)
                elif strategy == 'one':
                    p = self.masker.apply_one(base_mfr, byte_indices)
                elif strategy == 'rand':
                    p = self.masker.apply_rand(base_mfr, byte_indices, field)
                elif strategy == 'cross':
                    p = self.masker.apply_cross(base_mfr, byte_indices, true_label, sample_idx)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                perturbed_list.append(p)

            # Batch inference
            all_scores = []
            for i in range(0, len(perturbed_list), batch_size):
                batch = perturbed_list[i:i+batch_size]
                tensors = torch.cat([self._mfr_to_tensor(p) for p in batch], dim=0)
                scores = self._margin_score_batch(tensors, true_label)
                all_scores.append(scores)
            all_scores = np.concatenate(all_scores)

            # Attribution = original - mean(perturbed)
            results[strategy] = float(s_orig - all_scores.mean())

        return results

    def compute_coarse_attribution(
        self,
        mfr_flat: np.ndarray,
        true_label: int,
        sample_idx: int,
        region: str,  # 'header' or 'payload'
        strategies: List[str] = None,
        R: int = 50,
        batch_size: int = 50,
        effective_only: bool = False,
    ) -> Dict[str, float]:
        """
        Compute coarse-grained attribution for header or payload region.

        Args:
            region: 'header' or 'payload'.
            effective_only: If True, only mask effective (non-padding) bytes.
                           Padding bytes (trailing zeros) are left untouched.
        """
        if strategies is None:
            strategies = ['zero', 'one', 'rand', 'cross']

        if effective_only:
            if region == 'header':
                byte_indices = self.mapper.get_all_effective_header_indices(mfr_flat)
            else:
                byte_indices = self.mapper.get_all_effective_payload_indices(mfr_flat)
        else:
            if region == 'header':
                byte_indices = self.mapper.get_all_header_indices()
            else:
                byte_indices = self.mapper.get_all_payload_indices()

        orig_tensor = self._mfr_to_tensor(mfr_flat)
        s_orig = self._margin_score(orig_tensor, true_label)

        # For coarse attribution, use generic field for rand
        dummy_field = ProtocolField("Header_Region", "mixed", 0, 80, "mixed") if region == 'header' else PAYLOAD_FIELD

        results = {}
        for strategy in strategies:
            perturbed_list = []
            for _ in range(R):
                if strategy == 'zero':
                    p = self.masker.apply_zero(mfr_flat, byte_indices)
                elif strategy == 'one':
                    p = self.masker.apply_one(mfr_flat, byte_indices)
                elif strategy == 'rand':
                    # For coarse header masking, use uniform random
                    out = mfr_flat.copy()
                    out[byte_indices] = np.random.randint(0, 256, size=len(byte_indices), dtype=np.uint8)
                    p = out
                elif strategy == 'cross':
                    p = self.masker.apply_cross(mfr_flat, byte_indices, true_label, sample_idx)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                perturbed_list.append(p)

            all_scores = []
            for i in range(0, len(perturbed_list), batch_size):
                batch = perturbed_list[i:i+batch_size]
                tensors = torch.cat([self._mfr_to_tensor(p) for p in batch], dim=0)
                scores = self._margin_score_batch(tensors, true_label)
                all_scores.append(scores)
            all_scores = np.concatenate(all_scores)
            results[strategy] = float(s_orig - all_scores.mean())

        return results
