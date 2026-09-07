"""Threat classes and the horizons they select.

The correction these tests protect: harvest-now-decrypt-later is a
confidentiality threat, and scoring signatures against a data-retention horizon
conflated a 90-day leaf certificate with the 20-year root above it.
"""

import pytest

from qshield.algorithms import Primitive, primitive
from qshield.model import AssetModel, node_risk
from qshield.threat import (
    HORIZON_YEARS,
    MIN_LONGEVITY,
    ThreatClass,
    default_threat_class,
    longevity,
    rotation_is_mitigating,
)


@pytest.mark.parametrize(
    "algorithm,expected",
    [
        ("ECDSA", Primitive.SIGNATURE),
        ("ED25519", Primitive.SIGNATURE),
        ("ML-DSA", Primitive.SIGNATURE),
        ("SLH-DSA", Primitive.SIGNATURE),
        ("X25519", Primitive.KEY_ESTABLISHMENT),
        ("ML-KEM", Primitive.KEY_ESTABLISHMENT),
        ("RSA-2048", Primitive.KEY_ESTABLISHMENT),
        ("AES-256", Primitive.SYMMETRIC),
        ("SHA-256", Primitive.HASH),
    ],
)
def test_primitive_kind_is_registered(algorithm, expected):
    assert primitive(algorithm) is expected


def test_signatures_default_to_authentication_everything_else_to_confidentiality():
    assert default_threat_class("ECDSA") is ThreatClass.AUTHENTICATION
    assert default_threat_class("ML-DSA") is ThreatClass.AUTHENTICATION
    assert default_threat_class("X25519") is ThreatClass.CONFIDENTIALITY
    assert default_threat_class("AES-256") is ThreatClass.CONFIDENTIALITY


def test_each_class_reads_its_own_horizon():
    kwargs = dict(
        data_lifetime_years=20.0,
        credential_validity_years=2.0,
        verification_horizon_years=10.0,
    )
    assert longevity(ThreatClass.CONFIDENTIALITY, **kwargs) == pytest.approx(1.0)
    assert longevity(ThreatClass.AUTHENTICATION, **kwargs) == pytest.approx(0.1)
    assert longevity(ThreatClass.NON_REPUDIATION, **kwargs) == pytest.approx(0.5)


def test_missing_horizon_falls_back_to_data_lifetime():
    """Keeps pre-0.5 case files loadable -- and reproduces exactly the conflation
    this module exists to correct, which is why the fallback is documented."""
    value = longevity(
        ThreatClass.AUTHENTICATION,
        data_lifetime_years=10.0,
        credential_validity_years=None,
        verification_horizon_years=None,
    )
    assert value == pytest.approx(0.5)


def test_longevity_is_bounded():
    for horizon in (0.0, 0.01, 1.0, HORIZON_YEARS, 500.0):
        value = longevity(
            ThreatClass.CONFIDENTIALITY,
            data_lifetime_years=horizon,
            credential_validity_years=None,
            verification_horizon_years=None,
        )
        assert MIN_LONGEVITY <= value <= 1.0


def test_a_short_lived_leaf_and_a_long_lived_root_are_no_longer_the_same():
    """Both ECDSA, both fronting 15 years of data. Pre-0.5 scored them
    identically; the credential-validity horizon separates them tenfold."""
    shared = dict(sensitivity=0.8, exposure=0.8, data_lifetime_years=15.0,
                  threat_class=ThreatClass.AUTHENTICATION)
    leaf = AssetModel("leaf", "ECDSA", credential_validity_years=0.25, **shared)
    root = AssetModel("root", "ECDSA", credential_validity_years=20.0, **shared)
    assert node_risk(root) == pytest.approx(10 * node_risk(leaf))

    # Without the class-specific horizon they collapse back together.
    flat_leaf = AssetModel("leaf", "ECDSA", **shared)
    flat_root = AssetModel("root", "ECDSA", **shared)
    assert node_risk(flat_leaf) == pytest.approx(node_risk(flat_root))


def test_rotation_only_mitigates_authentication_risk():
    """Rotating a key does not un-harvest recorded ciphertext, and does not
    repair signed artifacts already in circulation."""
    assert rotation_is_mitigating(ThreatClass.AUTHENTICATION)
    assert not rotation_is_mitigating(ThreatClass.CONFIDENTIALITY)
    assert not rotation_is_mitigating(ThreatClass.NON_REPUDIATION)


def test_threat_class_is_an_explicit_override_not_an_inference():
    """A signing key whose artifacts must outlive it has to say so: nothing in
    the algorithm name distinguishes code signing from a login credential."""
    asset = AssetModel(
        "code-signing", "ECDSA", 0.9, 0.5, 5.0,
        threat_class=ThreatClass.NON_REPUDIATION, verification_horizon_years=25.0,
    )
    assert asset.effective_threat_class is ThreatClass.NON_REPUDIATION
    assert node_risk(asset) > node_risk(
        AssetModel("login", "ECDSA", 0.9, 0.5, 5.0, credential_validity_years=1.0)
    )


def test_negative_horizons_are_rejected():
    with pytest.raises(ValueError, match="credential_validity_years"):
        AssetModel("a", "ECDSA", credential_validity_years=-1)
    with pytest.raises(ValueError, match="verification_horizon_years"):
        AssetModel("a", "ECDSA", verification_horizon_years=-1)
