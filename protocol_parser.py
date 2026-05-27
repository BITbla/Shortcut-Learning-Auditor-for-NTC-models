"""
Protocol Structure Parser for YaTC MFR Matrix.

Maps MFR matrix byte positions to protocol fields, and provides
patch-to-field projection (phi) for PACA attribution aggregation.

MFR Layout (40x40 = 1600 bytes, 5 packets):
  Each packet = 8 rows x 40 cols = 320 bytes
    Row 0-1 (80 bytes): Header (IP + TCP/UDP, zero-padded to 80)
    Row 2-7 (240 bytes): Payload (first 240 bytes, zero-padded)

  Packet i starts at row i*8, byte offset i*320.

YaTC Patch: 2x2 pixels = 4 bytes. Grid: 20 cols x 4 rows per packet = 80 patches/packet.
  Total: 400 patches (but PatchEmbed sees img_size=(8,40) per packet -> 4x20=80 patches/packet).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class ProtocolField:
    name: str
    category: str          # 'shortcut', 'semantic', 'ambiguous', 'structure', 'padding'
    header_offset: int     # byte offset within the 80-byte header
    length: int            # in bytes
    expert_label: str      # 'known_shortcut', 'known_semantic', 'ambiguous', 'structure', 'padding'


# Standard IPv4 + TCP header fields within the 80-byte header region.
# IPv4 header: bytes 0-19 (standard 20 bytes, no options assumed)
# TCP header: bytes 20-39 (standard 20 bytes)
# Remaining: bytes 40-79 (zero-padded or options/extra header)

IP_TCP_FIELDS = [
    # IPv4 Header (20 bytes)
    ProtocolField("IP_Version_IHL",     "structure",  0,  1, "structure"),
    ProtocolField("IP_DSCP_ECN",        "structure",  1,  1, "structure"),
    ProtocolField("IP_Total_Length",     "semantic",   2,  2, "known_semantic"),
    ProtocolField("IP_ID",              "shortcut",   4,  2, "known_shortcut"),
    ProtocolField("IP_Flags_FragOff",   "structure",  6,  2, "structure"),
    ProtocolField("IP_TTL",             "ambiguous",  8,  1, "ambiguous"),
    ProtocolField("IP_Protocol",        "structure",  9,  1, "structure"),
    ProtocolField("IP_Checksum",        "shortcut",  10,  2, "known_shortcut"),
    ProtocolField("IP_Src",             "shortcut",  12,  4, "known_shortcut"),
    ProtocolField("IP_Dst",             "shortcut",  16,  4, "known_shortcut"),
    # TCP Header (20 bytes, starting at offset 20)
    ProtocolField("TCP_Src_Port",       "shortcut",  20,  2, "known_shortcut"),
    ProtocolField("TCP_Dst_Port",       "shortcut",  22,  2, "known_shortcut"),
    ProtocolField("TCP_Seq",            "shortcut",  24,  4, "known_shortcut"),
    ProtocolField("TCP_Ack",            "shortcut",  28,  4, "known_shortcut"),
    ProtocolField("TCP_DataOff_Flags",  "semantic",  32,  2, "known_semantic"),  # includes TCP Flags
    ProtocolField("TCP_Window",         "ambiguous",  34,  2, "ambiguous"),
    ProtocolField("TCP_Checksum",       "shortcut",  36,  2, "known_shortcut"),
    ProtocolField("TCP_Urgent",         "structure",  38,  2, "structure"),
    # TCP Options region (bytes 40-79, variable-length)
    # TCP Timestamp option (RFC 7323): kind=8, len=10, TSval(4B) + TSecr(4B)
    # Position is scanned dynamically; default offset 42 (after 2 NOP alignment bytes)
    ProtocolField("TCP_Timestamp",     "shortcut",  42,  8, "known_shortcut"),
    # Header padding / options region (remaining bytes)
    ProtocolField("Header_Padding",     "padding",   40, 40, "padding"),
]

# Dynamic fields that require per-packet position scanning
# TCP_Timestamp: scan for pattern (NOP|0x01, NOP|0x01, kind=8, len=10) in TCP options region
TIMESTAMP_OPTION_KIND = 8
TIMESTAMP_OPTION_LEN = 10
TIMESTAMP_DATA_LEN = 8  # TSval(4B) + TSecr(4B)
TCP_OPTIONS_START = 40  # byte offset where TCP options begin
TCP_OPTIONS_MAX_END = 78  # max scan end (leave room for option header + 8B data)

# Payload is treated as a single field for coarse analysis,
# but can be split into sub-regions for finer analysis.
PAYLOAD_FIELD = ProtocolField("Encrypted_Payload", "semantic", 80, 240, "known_semantic")

# All header fields that are NOT padding or structure
ATTRIBUTION_FIELDS = [f for f in IP_TCP_FIELDS if f.expert_label not in ('structure', 'padding')]

# Expert label sets for evaluation
KNOWN_SHORTCUT_FIELDS = {f.name for f in IP_TCP_FIELDS if f.expert_label == 'known_shortcut'}
KNOWN_SEMANTIC_FIELDS = {f.name for f in IP_TCP_FIELDS if f.expert_label == 'known_semantic'}
KNOWN_SEMANTIC_FIELDS.add("Encrypted_Payload")
AMBIGUOUS_FIELDS = {f.name for f in IP_TCP_FIELDS if f.expert_label == 'ambiguous'}


def get_all_fields(include_payload=True, include_padding=False, include_structure=False):
    """Return list of ProtocolField for attribution."""
    fields = []
    for f in IP_TCP_FIELDS:
        if f.expert_label == 'padding' and not include_padding:
            continue
        if f.expert_label == 'structure' and not include_structure:
            continue
        fields.append(f)
    if include_payload:
        fields.append(PAYLOAD_FIELD)
    return fields


class MFRProtocolMapper:
    """Maps MFR matrix positions to protocol fields for all 5 packets."""

    PACKETS = 5
    ROWS_PER_PACKET = 8
    COLS = 40
    BYTES_PER_PACKET = 320  # 8 * 40
    HEADER_BYTES = 80       # 2 rows * 40 cols
    PAYLOAD_BYTES = 240     # 6 rows * 40 cols
    PATCH_SIZE = 2
    PATCHES_PER_ROW = 20    # 40 / 2
    PATCH_ROWS_PER_PACKET = 4  # 8 / 2
    PATCHES_PER_PACKET = 80    # 4 * 20

    def __init__(self):
        self.fields = get_all_fields(include_payload=True)
        self._build_byte_to_field_map()
        self._build_patch_to_field_map()

    def _build_byte_to_field_map(self):
        """Map each byte within a packet's 320-byte block to a field name."""
        self.byte_to_field = {}  # (packet_idx, byte_in_packet) -> field_name
        for pkt in range(self.PACKETS):
            for f in IP_TCP_FIELDS:
                for b in range(f.header_offset, f.header_offset + f.length):
                    self.byte_to_field[(pkt, b)] = f.name
            # Payload bytes: offset 80-319
            for b in range(80, 320):
                self.byte_to_field[(pkt, b)] = "Encrypted_Payload"

    def _build_patch_to_field_map(self):
        """Map each patch index (0-399) to field(s) with fractional weights.

        Patch (pr, pc) in packet pkt covers:
          rows [pkt*8 + pr*2, pkt*8 + pr*2 + 1]
          cols [pc*2, pc*2 + 1]
          = 4 bytes at offsets within the packet.
        """
        self.patch_to_fields = {}  # patch_global_idx -> Dict[field_name, weight]

        for pkt in range(self.PACKETS):
            for pr in range(self.PATCH_ROWS_PER_PACKET):
                for pc in range(self.PATCHES_PER_ROW):
                    patch_idx = pkt * self.PATCHES_PER_PACKET + pr * self.PATCHES_PER_ROW + pc
                    field_weights = {}
                    for dr in range(2):
                        for dc in range(2):
                            row = pr * 2 + dr
                            col = pc * 2 + dc
                            byte_in_pkt = row * self.COLS + col
                            fname = self.byte_to_field.get((pkt, byte_in_pkt))
                            if fname:
                                field_weights[fname] = field_weights.get(fname, 0) + 0.25
                    self.patch_to_fields[patch_idx] = field_weights

    def get_field_byte_ranges(self, packet_idx: int, field: ProtocolField) -> List[Tuple[int, int]]:
        """Get (row, col) positions in the 40x40 MFR matrix for a field in a given packet."""
        positions = []
        base_row = packet_idx * self.ROWS_PER_PACKET
        for b in range(field.header_offset, field.header_offset + field.length):
            row = base_row + b // self.COLS
            col = b % self.COLS
            positions.append((row, col))
        return positions

    def get_field_byte_indices_flat(self, packet_idx: int, field: ProtocolField) -> List[int]:
        """Get flat byte indices (0-1599) in the MFR matrix for a field in a given packet."""
        base = packet_idx * self.BYTES_PER_PACKET
        return [base + b for b in range(field.header_offset, field.header_offset + field.length)]

    def get_header_byte_indices(self, packet_idx: int) -> List[int]:
        """Get flat byte indices for the entire header region of a packet."""
        base = packet_idx * self.BYTES_PER_PACKET
        return list(range(base, base + self.HEADER_BYTES))

    def get_payload_byte_indices(self, packet_idx: int) -> List[int]:
        """Get flat byte indices for the entire payload region of a packet."""
        base = packet_idx * self.BYTES_PER_PACKET
        return list(range(base + self.HEADER_BYTES, base + self.BYTES_PER_PACKET))

    def get_all_header_indices(self) -> List[int]:
        """Get flat byte indices for all header regions across all 5 packets."""
        indices = []
        for pkt in range(self.PACKETS):
            indices.extend(self.get_header_byte_indices(pkt))
        return indices

    def get_all_payload_indices(self) -> List[int]:
        """Get flat byte indices for all payload regions across all 5 packets."""
        indices = []
        for pkt in range(self.PACKETS):
            indices.extend(self.get_payload_byte_indices(pkt))
        return indices

    # ── Effective-byte-aware methods ──────────────────────────────────────

    @staticmethod
    def _effective_length(region_bytes: np.ndarray) -> int:
        """Estimate effective (non-padding) length by finding last non-zero byte.

        MFR uses 0x00 as padding, so trailing zeros are padding.
        Returns 0 if the entire region is zero (empty / padding packet).
        """
        nonzero = np.nonzero(region_bytes)[0]
        return int(nonzero[-1] + 1) if len(nonzero) > 0 else 0

    def get_effective_header_indices(self, packet_idx: int,
                                     mfr_flat: np.ndarray) -> List[int]:
        """Get flat byte indices for effective (non-padding) header bytes."""
        base = packet_idx * self.BYTES_PER_PACKET
        region = mfr_flat[base:base + self.HEADER_BYTES]
        eff_len = self._effective_length(region)
        return list(range(base, base + eff_len))

    def get_effective_payload_indices(self, packet_idx: int,
                                      mfr_flat: np.ndarray) -> List[int]:
        """Get flat byte indices for effective (non-padding) payload bytes."""
        base = packet_idx * self.BYTES_PER_PACKET + self.HEADER_BYTES
        region = mfr_flat[base:base + self.PAYLOAD_BYTES]
        eff_len = self._effective_length(region)
        return list(range(base, base + eff_len))

    def get_all_effective_header_indices(self, mfr_flat: np.ndarray) -> List[int]:
        """Get flat byte indices for all effective header bytes across 5 packets."""
        indices = []
        for pkt in range(self.PACKETS):
            indices.extend(self.get_effective_header_indices(pkt, mfr_flat))
        return indices

    def get_all_effective_payload_indices(self, mfr_flat: np.ndarray) -> List[int]:
        """Get flat byte indices for all effective payload bytes across 5 packets."""
        indices = []
        for pkt in range(self.PACKETS):
            indices.extend(self.get_effective_payload_indices(pkt, mfr_flat))
        return indices

    def get_field_indices_all_packets(self, field: ProtocolField,
                                       data: np.ndarray = None) -> List[int]:
        """Get flat byte indices for a field across all 5 packets.

        For fixed-offset fields, uses the field's header_offset and length.
        For dynamic-position fields (TCP_Timestamp), scans each packet for the
        timestamp option pattern using the provided data array.

        Args:
            field: The protocol field to locate.
            data: (1600,) uint8 MFR byte array. Required for dynamic fields.
        """
        if field.name == "TCP_Timestamp":
            if data is None:
                return []
            return self._get_timestamp_indices(data)
        indices = []
        for pkt in range(self.PACKETS):
            indices.extend(self.get_field_byte_indices_flat(pkt, field))
        return indices

    def _get_timestamp_indices(self, mfr_flat: np.ndarray) -> List[int]:
        """Scan all 5 packets in MFR data for TCP Timestamp data bytes.

        For each packet, scans the TCP options region (bytes 40-78) for the
        option header pattern (kind=8, len=10). Returns flat MFR indices for
        the 8 data bytes (TSval+TSecr) of packets where the pattern matches.
        Packets without a recognizable timestamp option are skipped.
        """
        indices = []
        for pkt in range(self.PACKETS):
            base = pkt * self.BYTES_PER_PACKET
            scan_end = min(self.HEADER_BYTES - 2 - TIMESTAMP_DATA_LEN,
                           TCP_OPTIONS_MAX_END)
            for offset in range(TCP_OPTIONS_START, scan_end + 1):
                pos = base + offset
                if (pos + 1 + TIMESTAMP_DATA_LEN <= base + self.HEADER_BYTES and
                    int(mfr_flat[pos]) == TIMESTAMP_OPTION_KIND and
                    int(mfr_flat[pos + 1]) == TIMESTAMP_OPTION_LEN):
                    data_start = pos + 2
                    indices.extend(range(data_start, data_start + TIMESTAMP_DATA_LEN))
                    break  # found in this packet, proceed to next
        return indices

    def find_timestamp_in_raw_bytes(self, raw_bytes: np.ndarray,
                                     bytes_per_packet: int = 320) -> List[int]:
        """Find TCP timestamp data byte indices in arbitrary raw packet bytes.

        Scans each packet for the timestamp option pattern (kind=8, len=10)
        and returns flat byte indices for TSval(4B)+TSecr(4B) across all
        packets where the option is present.

        Args:
            raw_bytes: Flat uint8 array of packet bytes.
            bytes_per_packet: Bytes per packet (320 for YaTC, 64 for TF).

        Returns:
            Flat byte indices for timestamp data bytes, or empty list.
        """
        indices = []
        n_packets = len(raw_bytes) // bytes_per_packet
        for pkt in range(n_packets):
            base = pkt * bytes_per_packet
            scan_end = min(bytes_per_packet - 2 - TIMESTAMP_DATA_LEN,
                           TCP_OPTIONS_MAX_END if bytes_per_packet > 80
                           else bytes_per_packet - 12)
            for offset in range(TCP_OPTIONS_START, scan_end + 1):
                pos = base + offset
                if (pos + 1 + TIMESTAMP_DATA_LEN <= len(raw_bytes) and
                    int(raw_bytes[pos]) == TIMESTAMP_OPTION_KIND and
                    int(raw_bytes[pos + 1]) == TIMESTAMP_OPTION_LEN):
                    data_start = pos + 2
                    indices.extend(range(data_start, data_start + TIMESTAMP_DATA_LEN))
                    break
        return indices

    def field_patch_weights(self, field_name: str) -> Dict[int, float]:
        """Get {patch_idx: weight} for patches that overlap with a given field."""
        result = {}
        for pidx, fweights in self.patch_to_fields.items():
            if field_name in fweights:
                result[pidx] = fweights[field_name]
        return result
