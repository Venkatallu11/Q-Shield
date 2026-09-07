"""Hybrid deployments, and the crossover they move.

A hybrid is broken only if both halves are, so it covers the CRQC and the
cryptanalysis of a young post-quantum primitive at once. That is what collapses
the crossover horizon the 0.7 calibration exposed.
"""

import pytest

from qshield.algorithms import (
    Family,
    Primitive,
    family,
    hybrid_components,
    normalize,
    primitive,
    quantum_factor,
)
from qshield.calibration import DEFAULT_CALIBRATION, CalibrationConfig, crossover_horizon_years
from qshield.hybrid import (
    HYBRID_SUITES,
    Deployment,
    classify_deployment,
    default_hybrid_for,
    hybrid_probability,
    migration_ladder,
    parse_hybrid,
)

# --- recognition -------------------------------------------------------------


def test_a_hybrid_is_recognised_from_its_components():
    parts = hybrid_components("X25519+ML-KEM")
    assert parts == ("X25519", "ML-KEM")


def test_component_order_does_not_matter():
    """Two spellings must name one thing, or the registry will hold duplicates."""
    assert normalize("ML-KEM+X25519") == normalize("X25519+ML-KEM") == "X25519+ML-KEM"


def test_a_hybrid_needs_one_classical_and_one_post_quantum_half():
    assert hybrid_components("RSA-2048+AES-256") is None   # no PQC half
    assert hybrid_components("ML-KEM+ML-DSA") is None      # no classical half
    assert hybrid_components("bogus+ML-KEM") is None       # unrecognised half
    assert hybrid_components("X25519") is None             # not a composition


def test_family_and_primitive_resolve_through_the_composition():
    assert family("X25519+ML-KEM") is Family.HYBRID
    assert family("ECDSA+ML-DSA") is Family.HYBRID
    # A hybrid does the job of its post-quantum half.
    assert primitive("X25519+ML-KEM") is Primitive.KEY_ESTABLISHMENT
    assert primitive("ECDSA+ML-DSA") is Primitive.SIGNATURE


def test_uncalibrated_factor_takes_the_stronger_half():
    """Broken only if both are, so on that scale a hybrid is no more susceptible
    than its better component."""
    assert quantum_factor("X25519+ML-KEM") == pytest.approx(quantum_factor("ML-KEM"))
    assert quantum_factor("X25519+ML-KEM") < quantum_factor("X25519")


def test_deployment_classification():
    assert classify_deployment("X25519+ML-KEM") is Deployment.HYBRID
    assert classify_deployment("X25519") is Deployment.CLASSICAL
    assert classify_deployment("ML-KEM") is Deployment.POST_QUANTUM
    assert classify_deployment("AES-256") is Deployment.SYMMETRIC
    assert classify_deployment("nonesuch") is Deployment.UNKNOWN


def test_shipped_suites_are_well_formed():
    for name, suite in HYBRID_SUITES.items():
        assert family(suite.classical) is Family.SHOR_VULNERABLE, name
        assert family(suite.post_quantum) is Family.POST_QUANTUM, name
        assert parse_hybrid(name) is not None, name


# --- composition -------------------------------------------------------------


def test_a_hybrid_is_broken_only_if_both_halves_are():
    suite = parse_hybrid("X25519+ML-KEM")
    combined = hybrid_probability(suite, 0.8, 0.05, composition_risk=0.0)
    assert combined == pytest.approx(0.8 * 0.05)


def test_composition_risk_is_a_floor_not_an_afterthought():
    """Gluing two key exchanges together can fail on its own; that is priced."""
    suite = parse_hybrid("X25519+ML-KEM")
    assert hybrid_probability(suite, 0.0, 0.0, composition_risk=0.02) == pytest.approx(0.02)
    assert hybrid_probability(suite, 1.0, 1.0, composition_risk=0.02) <= 1.0


def test_hybrid_beats_both_of_its_halves_at_every_horizon():
    config = DEFAULT_CALIBRATION
    for horizon in (0.25, 1, 5, 10, 20):
        hybrid = config.probability("X25519+ML-KEM", horizon)
        assert hybrid <= config.probability("X25519", horizon) + 1e-12
        assert hybrid <= config.probability("ML-KEM", horizon) + 1e-12


def test_hybrid_is_strictly_safer_than_pure_pqc_at_short_horizons():
    """The whole point: the classical half still protects you while the
    post-quantum half is young."""
    config = DEFAULT_CALIBRATION
    assert config.probability("X25519+ML-KEM", 0.25) < config.probability("ML-KEM", 0.25)


# --- the ladder --------------------------------------------------------------


def test_a_classical_primitive_offers_hybrid_then_pure():
    assert migration_ladder("X25519") == ("X25519+ML-KEM", "ML-KEM")
    assert migration_ladder("ECDSA") == ("ECDSA+ML-DSA", "ML-DSA")


def test_a_hybrid_can_still_advance_to_pure():
    assert migration_ladder("X25519+ML-KEM") == ("ML-KEM",)


def test_nothing_further_is_offered_from_the_top_or_from_symmetric():
    assert migration_ladder("ML-KEM") == ()
    assert migration_ladder("AES-256") == ()


def test_signature_primitives_get_a_signature_hybrid():
    assert default_hybrid_for("ECDSA") == "ECDSA+ML-DSA"
    assert default_hybrid_for("ED25519") == "ED25519+ML-DSA"
    assert default_hybrid_for("X25519") == "X25519+ML-KEM"


# --- the result that matters -------------------------------------------------


def test_hybrid_collapses_the_crossover_horizon():
    """0.7 said a 90-day certificate should be left alone because pure
    post-quantum only pays off above 3.7 years. Hybrid pays off at about eight
    weeks, which is shorter than the certificate's life."""
    config = DEFAULT_CALIBRATION
    to_pure = crossover_horizon_years("ECDSA", "ML-DSA", config=config)
    to_hybrid = crossover_horizon_years("ECDSA", "ECDSA+ML-DSA", config=config)
    assert to_pure > 3.0
    assert to_hybrid < 0.25          # shorter than a TLS leaf certificate's life
    assert to_hybrid < to_pure / 10


def test_the_crossover_gap_survives_a_pessimistic_composition_risk():
    """If gluing the halves together is itself risky, the advantage should
    shrink -- but it should not invert."""
    pessimistic = CalibrationConfig(composition_risk=0.04)
    to_pure = crossover_horizon_years("ECDSA", "ML-DSA", config=pessimistic)
    to_hybrid = crossover_horizon_years("ECDSA", "ECDSA+ML-DSA", config=pessimistic)
    assert to_hybrid < to_pure


def test_a_composition_risk_above_the_pqc_residual_removes_the_advantage():
    """The honest boundary: a hybrid is only better while its glue is safer than
    the primitive it is protecting."""
    reckless = CalibrationConfig(pqc_residual_risk=0.05, composition_risk=0.20)
    assert reckless.probability("X25519+ML-KEM", 1.0) > reckless.probability("ML-KEM", 1.0)
