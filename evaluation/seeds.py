from __future__ import annotations

import hashlib


def derive_seed(base_seed: int, label: str) -> int:
    """Derive a stable uint32 seed from a frozen base seed and semantic label."""
    payload = f"finauth-audit-v0.2.0/{base_seed}/{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
