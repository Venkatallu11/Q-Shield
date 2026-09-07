"""Single source of truth for cryptographic algorithm classification.

Q-SHIELD 0.3 carried two independent, silently divergent taxonomies
(``engine.quantum_factor`` and ``model.QF``). They disagreed on common inputs:
``RSA`` scored 1.0 in one and 0.5 (the "unknown" default) in the other, and the
aliases ``Kyber``/``Dilithium``/``SPHINCS+`` resolved in one module only. Because
the objective is a product of factors, a silent 0.5 halves or doubles an asset's
modeled risk without any diagnostic. This module replaces both.

The quantum factors below are *scenario parameters*, not empirical probabilities.
They encode a single ordering hypothesis: primitives broken by a
cryptographically relevant quantum computer (CRQC) via Shor's algorithm dominate
symmetric primitives weakened only by Grover-style search, which in turn dominate
NIST-selected post-quantum primitives. Magnitudes are placeholders awaiting
calibration; see docs/METHODOLOGY.md.
"""

from __future__ import annotations

from enum import Enum

# Sentinel returned for algorithms the registry does not recognise.
UNKNOWN_QUANTUM_FACTOR = 0.5

ROTATION_REFERENCE_HOURS = 168.0  # one week; normalises the rotation-pressure term


class Family(str, Enum):
    """Coarse classification used for reporting and CBOM export."""

    SHOR_VULNERABLE = "shor-vulnerable"      # public-key, broken by a CRQC
    GROVER_WEAKENED = "grover-weakened"      # symmetric/hash, effective strength halved
    POST_QUANTUM = "post-quantum"            # NIST PQC selections
    UNKNOWN = "unknown"


# canonical name -> (family, quantum factor)
_REGISTRY: dict[str, tuple[Family, float]] = {
    # --- Shor-vulnerable public-key primitives -------------------------------
    "RSA": (Family.SHOR_VULNERABLE, 1.00),
    "RSA-1024": (Family.SHOR_VULNERABLE, 1.00),
    "RSA-2048": (Family.SHOR_VULNERABLE, 1.00),
    "RSA-3072": (Family.SHOR_VULNERABLE, 1.00),
    "RSA-4096": (Family.SHOR_VULNERABLE, 1.00),
    "ECC": (Family.SHOR_VULNERABLE, 1.00),
    "ECDSA": (Family.SHOR_VULNERABLE, 1.00),
    "ECDH": (Family.SHOR_VULNERABLE, 1.00),
    "X25519": (Family.SHOR_VULNERABLE, 1.00),
    "ED25519": (Family.SHOR_VULNERABLE, 1.00),
    "ED448": (Family.SHOR_VULNERABLE, 1.00),
    "DH": (Family.SHOR_VULNERABLE, 1.00),
    "DSA": (Family.SHOR_VULNERABLE, 1.00),
    # --- NIST post-quantum selections ----------------------------------------
    "ML-KEM": (Family.POST_QUANTUM, 0.05),
    "ML-DSA": (Family.POST_QUANTUM, 0.05),
    "SLH-DSA": (Family.POST_QUANTUM, 0.05),
    "FN-DSA": (Family.POST_QUANTUM, 0.05),
    "HQC": (Family.POST_QUANTUM, 0.05),
    # --- Symmetric / hash primitives -----------------------------------------
    "AES-128": (Family.GROVER_WEAKENED, 0.15),
    "AES-256": (Family.GROVER_WEAKENED, 0.08),
    "SHA-256": (Family.GROVER_WEAKENED, 0.08),
    "SHA-384": (Family.GROVER_WEAKENED, 0.08),
    "SHA-512": (Family.GROVER_WEAKENED, 0.08),
    "CHACHA20": (Family.GROVER_WEAKENED, 0.08),
}

# Spelling variants and pre-standardisation names -> canonical registry key.
_ALIASES: dict[str, str] = {
    "KYBER": "ML-KEM",
    "CRYSTALS-KYBER": "ML-KEM",
    "DILITHIUM": "ML-DSA",
    "CRYSTALS-DILITHIUM": "ML-DSA",
    "SPHINCS": "SLH-DSA",
    "SPHINCS+": "SLH-DSA",
    "FALCON": "FN-DSA",
    "CURVE25519": "X25519",
    "ECDSA-P256": "ECDSA",
    "ECDSA-P384": "ECDSA",
    "SECP256R1": "ECDH",
    "AES": "AES-128",  # conservative: assume the weaker parameter set
}


class UnknownAlgorithmError(KeyError):
    """Raised in strict mode when an algorithm is absent from the registry."""


def normalize(algorithm: str) -> str:
    """Canonicalise an algorithm string.

    Case, surrounding whitespace and ``_``/``-`` separator choice are all
    insignificant. Alias resolution happens here so every caller agrees.
    """
    key = algorithm.strip().upper().replace("_", "-")
    while "--" in key:
        key = key.replace("--", "-")
    return _ALIASES.get(key, key)


def is_known(algorithm: str) -> bool:
    return normalize(algorithm) in _REGISTRY


def family(algorithm: str) -> Family:
    entry = _REGISTRY.get(normalize(algorithm))
    return entry[0] if entry else Family.UNKNOWN


def quantum_factor(algorithm: str, *, strict: bool = False) -> float:
    """Scenario susceptibility factor in ``(0, 1]``.

    With ``strict=True`` an unrecognised algorithm raises instead of silently
    taking :data:`UNKNOWN_QUANTUM_FACTOR`. Experiments run strict by default so a
    typo in an input file fails loudly rather than perturbing every downstream
    number by a factor of two.
    """
    entry = _REGISTRY.get(normalize(algorithm))
    if entry is not None:
        return entry[1]
    if strict:
        raise UnknownAlgorithmError(
            f"algorithm {algorithm!r} (normalised {normalize(algorithm)!r}) is not in "
            f"the registry; add it to qshield.algorithms or pass strict=False to "
            f"accept the {UNKNOWN_QUANTUM_FACTOR} placeholder"
        )
    return UNKNOWN_QUANTUM_FACTOR


def default_replacement(algorithm: str) -> str | None:
    """Conventional post-quantum successor for a Shor-vulnerable primitive.

    Key-establishment primitives map to ML-KEM, signature primitives to ML-DSA.
    Returns ``None`` when no migration is defined (already post-quantum, or a
    symmetric primitive whose remedy is a parameter change, not a swap).
    """
    key = normalize(algorithm)
    if family(key) is not Family.SHOR_VULNERABLE:
        return None
    signatures = {"ECDSA", "ED25519", "ED448", "DSA"}
    if key in signatures or key.startswith("RSA-PSS"):
        return "ML-DSA"
    if key.startswith("RSA"):
        # RSA is used for both; key transport is the dominant migration driver.
        return "ML-KEM"
    return "ML-KEM"


def registry_snapshot() -> dict[str, dict[str, object]]:
    """Machine-readable dump of the registry, for embedding in result files."""
    return {
        name: {"family": fam.value, "quantum_factor": qf}
        for name, (fam, qf) in sorted(_REGISTRY.items())
    }
