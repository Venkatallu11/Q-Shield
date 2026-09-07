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
    HYBRID = "hybrid"                        # classical + PQC, broken only if both are
    UNKNOWN = "unknown"


class Primitive(str, Enum):
    """What the algorithm is *for*.

    This drives the threat class an asset defaults to, and that in turn selects
    which time horizon its risk is measured against. Confidentiality primitives
    are exposed to harvest-now-decrypt-later; signature primitives are not, and
    scoring them against a data-retention horizon — which every version through
    0.4 did — overstates the risk of short-lived credentials and understates
    nothing in return. See :mod:`qshield.threat`.
    """

    KEY_ESTABLISHMENT = "key-establishment"  # KEM / key agreement / key transport
    SIGNATURE = "signature"
    SYMMETRIC = "symmetric"                  # bulk encryption
    HASH = "hash"
    UNKNOWN = "unknown"


# canonical name -> (family, quantum factor, primitive kind)
_REGISTRY: dict[str, tuple[Family, float, Primitive]] = {
    # --- Shor-vulnerable public-key primitives -------------------------------
    # RSA is used for both key transport and signatures. It is registered as
    # key-establishment because that is the migration driver (harvested key
    # exchanges are decryptable retroactively); an asset that uses RSA to sign
    # should declare threat_class=AUTHENTICATION or NON_REPUDIATION explicitly.
    "RSA": (Family.SHOR_VULNERABLE, 1.00, Primitive.KEY_ESTABLISHMENT),
    "RSA-1024": (Family.SHOR_VULNERABLE, 1.00, Primitive.KEY_ESTABLISHMENT),
    "RSA-2048": (Family.SHOR_VULNERABLE, 1.00, Primitive.KEY_ESTABLISHMENT),
    "RSA-3072": (Family.SHOR_VULNERABLE, 1.00, Primitive.KEY_ESTABLISHMENT),
    "RSA-4096": (Family.SHOR_VULNERABLE, 1.00, Primitive.KEY_ESTABLISHMENT),
    "ECC": (Family.SHOR_VULNERABLE, 1.00, Primitive.KEY_ESTABLISHMENT),
    "ECDH": (Family.SHOR_VULNERABLE, 1.00, Primitive.KEY_ESTABLISHMENT),
    "X25519": (Family.SHOR_VULNERABLE, 1.00, Primitive.KEY_ESTABLISHMENT),
    "DH": (Family.SHOR_VULNERABLE, 1.00, Primitive.KEY_ESTABLISHMENT),
    "ECDSA": (Family.SHOR_VULNERABLE, 1.00, Primitive.SIGNATURE),
    "ED25519": (Family.SHOR_VULNERABLE, 1.00, Primitive.SIGNATURE),
    "ED448": (Family.SHOR_VULNERABLE, 1.00, Primitive.SIGNATURE),
    "DSA": (Family.SHOR_VULNERABLE, 1.00, Primitive.SIGNATURE),
    # --- NIST post-quantum selections ----------------------------------------
    "ML-KEM": (Family.POST_QUANTUM, 0.05, Primitive.KEY_ESTABLISHMENT),
    "HQC": (Family.POST_QUANTUM, 0.05, Primitive.KEY_ESTABLISHMENT),
    "ML-DSA": (Family.POST_QUANTUM, 0.05, Primitive.SIGNATURE),
    "SLH-DSA": (Family.POST_QUANTUM, 0.05, Primitive.SIGNATURE),
    "FN-DSA": (Family.POST_QUANTUM, 0.05, Primitive.SIGNATURE),
    # --- Symmetric / hash primitives -----------------------------------------
    "AES-128": (Family.GROVER_WEAKENED, 0.15, Primitive.SYMMETRIC),
    "AES-256": (Family.GROVER_WEAKENED, 0.08, Primitive.SYMMETRIC),
    "CHACHA20": (Family.GROVER_WEAKENED, 0.08, Primitive.SYMMETRIC),
    "SHA-256": (Family.GROVER_WEAKENED, 0.08, Primitive.HASH),
    "SHA-384": (Family.GROVER_WEAKENED, 0.08, Primitive.HASH),
    "SHA-512": (Family.GROVER_WEAKENED, 0.08, Primitive.HASH),
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


HYBRID_SEPARATOR = "+"


def hybrid_components(algorithm: str) -> tuple[str, str] | None:
    """Split ``classical+post-quantum`` into its halves, or return None.

    Lives here rather than in :mod:`qshield.hybrid` so that ``family``,
    ``primitive`` and ``quantum_factor`` can recognise a hybrid without importing
    that module, which would be circular. The deployed-suite catalogue and the
    probability composition stay there.
    """
    key = algorithm.strip().upper().replace("_", "-")
    if HYBRID_SEPARATOR not in key:
        return None
    left, _, right = key.partition(HYBRID_SEPARATOR)
    left, right = normalize(left), normalize(right)
    entries = (_REGISTRY.get(left), _REGISTRY.get(right))
    if None in entries:
        return None
    families = [entry[0] for entry in entries]
    if Family.SHOR_VULNERABLE not in families or Family.POST_QUANTUM not in families:
        return None
    classical = left if families[0] is Family.SHOR_VULNERABLE else right
    post_quantum = right if classical == left else left
    return classical, post_quantum


def normalize(algorithm: str) -> str:
    """Canonicalise an algorithm string.

    Case, surrounding whitespace and ``_``/``-`` separator choice are all
    insignificant. Alias resolution happens here so every caller agrees. A
    hybrid is canonicalised to ``CLASSICAL+POST-QUANTUM`` with both halves
    resolved and the classical half first, so the two orderings name one thing.
    """
    key = algorithm.strip().upper().replace("_", "-")
    if HYBRID_SEPARATOR in key:
        parts = hybrid_components(key)
        if parts is not None:
            return f"{parts[0]}{HYBRID_SEPARATOR}{parts[1]}"
    while "--" in key:
        key = key.replace("--", "-")
    return _ALIASES.get(key, key)


def is_known(algorithm: str) -> bool:
    return normalize(algorithm) in _REGISTRY


def family(algorithm: str) -> Family:
    if hybrid_components(algorithm) is not None:
        return Family.HYBRID
    entry = _REGISTRY.get(normalize(algorithm))
    return entry[0] if entry else Family.UNKNOWN


def primitive(algorithm: str) -> Primitive:
    """What the algorithm is used for. Drives the default threat class."""
    parts = hybrid_components(algorithm)
    if parts is not None:
        # A hybrid does the job of its post-quantum half.
        return primitive(parts[1])
    entry = _REGISTRY.get(normalize(algorithm))
    return entry[2] if entry else Primitive.UNKNOWN


def quantum_factor(algorithm: str, *, strict: bool = False) -> float:
    """Scenario susceptibility factor in ``(0, 1]``.

    With ``strict=True`` an unrecognised algorithm raises instead of silently
    taking :data:`UNKNOWN_QUANTUM_FACTOR`. Experiments run strict by default so a
    typo in an input file fails loudly rather than perturbing every downstream
    number by a factor of two.
    """
    parts = hybrid_components(algorithm)
    if parts is not None:
        # Broken only if both halves are, so on this uncalibrated scale a hybrid
        # is no more susceptible than its stronger half. The calibrated path in
        # qshield.hybrid composes the two probabilities properly.
        return min(quantum_factor(parts[0]), quantum_factor(parts[1]))
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


def default_replacement(algorithm: str) -> str | None:  # noqa: D401
    """Conventional post-quantum successor for a Shor-vulnerable primitive.

    Key-establishment primitives map to ML-KEM, signature primitives to ML-DSA.
    Returns ``None`` when no migration is defined (already post-quantum, or a
    symmetric primitive whose remedy is a parameter change, not a swap).
    """
    key = normalize(algorithm)
    if family(key) is not Family.SHOR_VULNERABLE:
        return None
    if primitive(key) is Primitive.SIGNATURE:
        return "ML-DSA"
    return "ML-KEM"


def registry_snapshot() -> dict[str, dict[str, object]]:
    """Machine-readable dump of the registry, for embedding in result files."""
    return {
        name: {"family": fam.value, "quantum_factor": qf, "primitive": prim.value}
        for name, (fam, qf, prim) in sorted(_REGISTRY.items())
    }
