"""The registry is the single source of truth; 0.3 had two that disagreed."""

import pytest

from qshield.algorithms import (
    UNKNOWN_QUANTUM_FACTOR,
    Family,
    UnknownAlgorithmError,
    default_replacement,
    is_known,
    normalize,
    quantum_factor,
    registry_snapshot,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("rsa-2048", "RSA-2048"),
        ("  RSA_2048 ", "RSA-2048"),
        ("kyber", "ML-KEM"),
        ("CRYSTALS-Dilithium", "ML-DSA"),
        ("sphincs+", "SLH-DSA"),
        ("falcon", "FN-DSA"),
        ("sha_256", "SHA-256"),
        ("ed25519", "ED25519"),
    ],
)
def test_normalize_resolves_case_separator_and_alias(raw, expected):
    assert normalize(raw) == expected


def test_algorithms_0_3_scored_as_unknown_are_now_classified():
    # Each of these hit model.QF's 0.5 default in 0.3 while engine.py knew them,
    # so the same asset scored differently depending on the code path taken.
    for alg in ("RSA", "Kyber", "Dilithium", "SPHINCS+", "SHA-256", "FN-DSA"):
        assert is_known(alg), alg
        assert quantum_factor(alg) != UNKNOWN_QUANTUM_FACTOR, alg


def test_ordering_hypothesis_holds_across_families():
    assert quantum_factor("RSA-2048") > quantum_factor("AES-128")
    assert quantum_factor("AES-128") > quantum_factor("AES-256")
    assert quantum_factor("AES-256") > quantum_factor("ML-KEM")


def test_unknown_algorithm_is_silent_by_default_and_loud_in_strict_mode():
    assert quantum_factor("nonesuch") == UNKNOWN_QUANTUM_FACTOR
    with pytest.raises(UnknownAlgorithmError):
        quantum_factor("nonesuch", strict=True)


def test_families_partition_the_registry():
    for name, entry in registry_snapshot().items():
        assert entry["family"] != Family.UNKNOWN.value, name
        assert 0.0 < entry["quantum_factor"] <= 1.0, name


def test_default_replacement_targets_the_right_primitive_kind():
    assert default_replacement("ECDSA") == "ML-DSA"
    assert default_replacement("ED25519") == "ML-DSA"
    assert default_replacement("X25519") == "ML-KEM"
    assert default_replacement("RSA-2048") == "ML-KEM"
    # Already post-quantum, or symmetric: no swap is defined.
    assert default_replacement("ML-KEM") is None
    assert default_replacement("AES-256") is None


def test_replacements_strictly_reduce_susceptibility():
    for alg in ("RSA-2048", "ECDSA", "X25519", "ED25519", "DH", "DSA"):
        assert quantum_factor(default_replacement(alg)) < quantum_factor(alg)
