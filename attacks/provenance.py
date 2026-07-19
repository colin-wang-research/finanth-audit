from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


ELIGIBLE_ROLES = ("edge_proposer", "learned_prior", "policy_proposer")
INELIGIBLE_ROLES = ("risk_critic", "reasoning_prior", "adversarial_reasoning_prior")
ALL_ROLES = ELIGIBLE_ROLES + INELIGIBLE_ROLES


@dataclass(frozen=True)
class ProvenanceTransform:
    attack_type: str
    original_source_id: str
    current_source_id: str
    original_verified_role: str
    verified_current_role: str
    claimed_role: str
    current_role_verified: bool
    delegated_from: str
    transformation_chain: str
    evidence_lineage: str
    lineage_role_chain: str
    hop_depth: int
    transformation_type: str
    traceability: str
    lineage_attested: bool
    source_content_hash: str
    role_noise_rate: float
    role_noise_applied: bool
    original_source_eligible: bool
    current_source_eligible: bool
    authority_laundering: bool
    safe_delegation: bool
    detectable_in_principle: bool


def role_eligible(role: str) -> bool:
    return role in ELIGIBLE_ROLES


def _content_hash(row_id: str, original: str, current: str, chain: list[dict[str, str]]) -> str:
    payload = json.dumps(
        {"row_id": row_id, "original": original, "current": current, "chain": chain},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _choose_eligible(rng: np.random.Generator) -> str:
    return str(rng.choice(ELIGIBLE_ROLES))


def transform(
    *,
    row_id: str,
    original_source_id: str,
    original_role: str,
    attack_type: str,
    traceability: str,
    role_noise_rate: float,
    hop_depth: int,
    rng: np.random.Generator,
) -> ProvenanceTransform:
    if attack_type not in {"clean", "role_noise", "delegation", "paraphrase", "multi_hop"}:
        raise ValueError(f"unknown attack {attack_type}")
    if traceability not in {"traceable", "untraceable"}:
        raise ValueError(f"unknown traceability {traceability}")

    original_eligible = role_eligible(original_role)
    current_role = original_role
    current_source_id = original_source_id
    claimed_role = original_role
    current_role_verified = True
    role_noise_applied = False
    delegated_from = ""
    chain: list[dict[str, str]] = [
        {"hop": "0", "source": original_source_id, "role": original_role, "operation": "origin"}
    ]
    transformation_type = "none"
    effective_hops = 0

    if attack_type == "role_noise":
        role_noise_applied = bool(rng.random() < role_noise_rate)
        if role_noise_applied:
            claimed_role = _choose_eligible(rng) if not original_eligible else str(rng.choice(INELIGIBLE_ROLES))
            current_role_verified = False
        chain.append(
            {
                "hop": "1",
                "source": original_source_id,
                "role": claimed_role,
                "operation": "role_metadata_update",
            }
        )
        transformation_type = "role_noise"
        effective_hops = 1
    elif attack_type in {"delegation", "paraphrase"}:
        current_role = _choose_eligible(rng)
        current_source_id = f"{attack_type}-receiver-{rng.integers(0, 10_000):04d}"
        claimed_role = current_role
        delegated_from = original_source_id
        transformation_type = "delegate" if attack_type == "delegation" else "paraphrase"
        effective_hops = 1
        chain.append(
            {
                "hop": "1",
                "source": current_source_id,
                "role": current_role,
                "operation": transformation_type,
            }
        )
    elif attack_type == "multi_hop":
        effective_hops = max(1, min(5, int(hop_depth)))
        delegated_from = original_source_id
        for hop in range(1, effective_hops + 1):
            is_final = hop == effective_hops
            if is_final:
                role = _choose_eligible(rng)
            elif original_eligible:
                role = _choose_eligible(rng)
            else:
                role = str(rng.choice(ALL_ROLES))
            source = f"hop-{hop}-source-{rng.integers(0, 10_000):04d}"
            operation = str(rng.choice(("summarize", "rewrite", "delegate")))
            chain.append({"hop": str(hop), "source": source, "role": role, "operation": operation})
        current_source_id = chain[-1]["source"]
        current_role = chain[-1]["role"]
        claimed_role = current_role
        transformation_type = chain[-1]["operation"]

    current_eligible = role_eligible(current_role)
    upstream_roles = [item["role"] for item in chain[:-1]] if len(chain) > 1 else [original_role]
    authority_laundering = current_eligible and any(not role_eligible(role) for role in upstream_roles)
    safe_delegation = attack_type in {"delegation", "paraphrase", "multi_hop"} and current_eligible and all(
        role_eligible(item["role"]) for item in chain
    )
    lineage_attested = traceability == "traceable"
    evidence_lineage = json.dumps(chain, sort_keys=True) if lineage_attested else ""
    lineage_role_chain = "|".join(item["role"] for item in chain) if lineage_attested else ""

    return ProvenanceTransform(
        attack_type=attack_type,
        original_source_id=original_source_id,
        current_source_id=current_source_id,
        original_verified_role=original_role,
        verified_current_role=current_role,
        claimed_role=claimed_role,
        current_role_verified=current_role_verified,
        delegated_from=delegated_from,
        transformation_chain=json.dumps(chain, sort_keys=True),
        evidence_lineage=evidence_lineage,
        lineage_role_chain=lineage_role_chain,
        hop_depth=effective_hops,
        transformation_type=transformation_type,
        traceability=traceability,
        lineage_attested=lineage_attested,
        source_content_hash=_content_hash(row_id, original_source_id, current_source_id, chain),
        role_noise_rate=float(role_noise_rate),
        role_noise_applied=role_noise_applied,
        original_source_eligible=original_eligible,
        current_source_eligible=current_eligible,
        authority_laundering=authority_laundering,
        safe_delegation=safe_delegation,
        detectable_in_principle=lineage_attested or attack_type in {"clean", "role_noise"},
    )
