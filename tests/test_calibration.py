"""Calibration: published estimates in place of invented constants.

These tests pin the two places where the literature disagrees with the registry
shipped through 0.6, and the behaviour that disagreement produces.
"""

import math

import pytest

from qshield.calibration import (
    DEFAULT_CALIBRATION,
    EXPERT_FORECASTS,
    PQC_RESIDUAL_RISK,
    QUANTUM_RESOURCES,
    SYMMETRIC_CAPABILITY_PROBABILITY,
    CalibrationConfig,
    CRQCForecast,
    calibrated_quantum_factor,
    capability_probability,
    crossover_horizon_years,
    horizon_for,
    lead_time_years,
    migration_is_net_positive,
    mosca_verdict,
    provenance,
    resource_for,
    years_to_confidence,
)
from qshield.model import AssetModel, node_risk
from qshield.threat import ThreatClass

# --- Tier A(i): relative quantum work ---------------------------------------


def test_elliptic_curve_is_an_easier_target_than_rsa_at_equal_security():
    """The correction the shipped registry got backwards.

    Roetteler et al. put ECC-256 at 2330 logical qubits and 1.26e11 Toffoli gates
    against RSA-3072's 6146 and 1.86e13 -- both at ~128-bit classical security.
    The registry scored both at exactly 1.00, asserting they fall together.
    """
    ecc = QUANTUM_RESOURCES["ECDSA"]
    rsa = QUANTUM_RESOURCES["RSA-3072"]
    assert ecc.classical_security_bits == rsa.classical_security_bits
    assert ecc.logical_qubits < rsa.logical_qubits
    assert ecc.toffoli_gates < rsa.toffoli_gates
    assert rsa.toffoli_gates / ecc.toffoli_gates > 100


def test_curve_variants_share_the_ecdlp_cost():
    for algorithm in ("ECDH", "X25519", "ED25519", "ECC"):
        assert resource_for(algorithm) is QUANTUM_RESOURCES["ECDSA"]


def test_elliptic_curve_falls_before_rsa():
    assert lead_time_years("ECDSA") > 0
    assert lead_time_years("X25519") > 0
    # RSA-2048 is the anchor, so it leads itself by nothing.
    assert lead_time_years("RSA-2048") == pytest.approx(0.0)
    # A larger modulus is harder, so it falls later.
    assert lead_time_years("RSA-3072") < 0


def test_lead_time_scales_with_the_hardware_assumption():
    """Tier B: the resource ratio is published, the doubling time is not."""
    slow = lead_time_years("ECDSA", doubling_time_years=4.0)
    fast = lead_time_years("ECDSA", doubling_time_years=1.0)
    assert slow == pytest.approx(4 * fast)


def test_every_resource_entry_cites_a_source():
    for name, resource in QUANTUM_RESOURCES.items():
        assert resource.source, name
        assert resource.logical_qubits > 0 and resource.toffoli_gates > 0


# --- Tier A(ii): when the capability arrives --------------------------------


def test_forecast_is_a_monotone_cdf():
    for forecast in EXPERT_FORECASTS.values():
        previous = 0.0
        for years in (0, 1, 5, 10, 15, 20, 30, 50):
            value = forecast.probability_within(years)
            assert 0.0 <= value <= 1.0
            assert value >= previous - 1e-12, (forecast.name, years)
            previous = value


def test_forecast_anchors_match_the_published_survey():
    """GRI/evolutionQ 2025: 28-49% within 10 years, 51-70% within 15."""
    assert EXPERT_FORECASTS["conservative"].probability_within(10) == pytest.approx(0.28)
    assert EXPERT_FORECASTS["aggressive"].probability_within(10) == pytest.approx(0.49)
    assert EXPERT_FORECASTS["conservative"].probability_within(15) == pytest.approx(0.51)
    assert EXPERT_FORECASTS["aggressive"].probability_within(15) == pytest.approx(0.70)


def test_scenarios_are_ordered_everywhere():
    for years in (5, 10, 15, 20, 30):
        low = EXPERT_FORECASTS["conservative"].probability_within(years)
        mid = EXPERT_FORECASTS["central"].probability_within(years)
        high = EXPERT_FORECASTS["aggressive"].probability_within(years)
        assert low <= mid <= high


def test_forecast_is_held_flat_past_its_last_anchor():
    """Extrapolating an elicitation past the horizon experts were asked about
    would invent data."""
    forecast = EXPERT_FORECASTS["central"]
    assert forecast.probability_within(100) == forecast.probability_within(30)


def test_zero_horizon_carries_no_probability():
    assert capability_probability("RSA-2048", 0) == 0.0
    assert capability_probability("RSA-2048", -5) == 0.0


# --- The symmetric correction -----------------------------------------------


def test_symmetric_primitives_carry_no_quantum_capability_risk():
    """NIST makes AES-128 the *benchmark* for post-quantum security, and key
    search on AES-256 costs about 2^298/MAXDEPTH gates. The shipped registry
    scored AES-128 at 0.15 and AES-256 at 0.08 -- above the 0.05 it gave ML-KEM,
    so the model rated symmetric cryptography riskier than the post-quantum
    algorithms it recommends migrating to."""
    for algorithm in ("AES-128", "AES-256", "SHA-256", "CHACHA20"):
        assert capability_probability(algorithm, 30) == SYMMETRIC_CAPABILITY_PROBABILITY
    from qshield.algorithms import quantum_factor

    assert quantum_factor("AES-128") > quantum_factor("ML-KEM")  # the old ordering
    assert capability_probability("AES-128", 30) < capability_probability("ML-KEM", 30)


def test_symmetric_never_reaches_any_confidence_level():
    assert years_to_confidence("AES-256", 0.01) == math.inf


def test_post_quantum_carries_a_residual_that_is_not_a_quantum_estimate():
    assert capability_probability("ML-KEM", 1) == PQC_RESIDUAL_RISK
    assert capability_probability("ML-KEM", 30) == PQC_RESIDUAL_RISK
    assert "not a quantum-susceptibility estimate" in provenance()[
        "tier_c_unobservable"
    ]["pqc_residual_risk"]["source"]


def test_an_unknown_primitive_is_not_assumed_safe():
    """Guessing low is the dangerous direction."""
    assert capability_probability("something-unregistered", 20) > 0.5


# --- The crossover: migration is not always net-positive --------------------


def test_short_lived_credentials_are_not_worth_migrating():
    """The finding the uncalibrated model could not express: it multiplied a
    quantum factor of 1.00 by a longevity term, so migration always helped."""
    assert not migration_is_net_positive("ECDSA", "ML-DSA", 0.25)  # a 90-day cert
    assert migration_is_net_positive("ECDSA", "ML-DSA", 20.0)


def test_crossover_is_ordered_by_how_easy_the_target_is():
    """ECC falls sooner, so migrating it pays off at a shorter horizon."""
    ecc = crossover_horizon_years("ECDSA", "ML-DSA")
    rsa = crossover_horizon_years("RSA-2048", "ML-KEM")
    assert 0 < ecc < rsa


def test_crossover_location_depends_on_the_uncalibrated_residual():
    """Its existence follows from the calibrated form; its location does not."""
    low = crossover_horizon_years(
        "ECDSA", "ML-DSA", config=CalibrationConfig(pqc_residual_risk=0.01)
    )
    high = crossover_horizon_years(
        "ECDSA", "ML-DSA", config=CalibrationConfig(pqc_residual_risk=0.10)
    )
    assert low < high
    assert high / max(low, 1e-9) > 2


def test_symmetric_migration_never_pays():
    assert crossover_horizon_years("AES-256", "ML-KEM") == math.inf


# --- Mosca's inequality -----------------------------------------------------


def test_mosca_flags_a_migration_that_started_too_late():
    verdict = mosca_verdict(shelf_life_years=15, migration_years=5)
    assert verdict.deadline_missed
    assert verdict.slack_years < 0


def test_mosca_passes_a_short_lived_asset():
    verdict = mosca_verdict(shelf_life_years=1, migration_years=1)
    assert not verdict.deadline_missed
    assert verdict.slack_years > 0


def test_mosca_confidence_is_explicit_not_hidden():
    cautious = mosca_verdict(shelf_life_years=5, migration_years=2, confidence=0.1)
    relaxed = mosca_verdict(shelf_life_years=5, migration_years=2, confidence=0.9)
    assert cautious.years_until_capability < relaxed.years_until_capability
    assert cautious.as_dict()["at_confidence"] == 0.1


def test_mosca_on_a_symmetric_primitive_never_expires():
    verdict = mosca_verdict(
        shelf_life_years=100, migration_years=100, algorithm="AES-256"
    )
    assert verdict.years_until_capability == math.inf
    assert not verdict.deadline_missed


# --- Wiring into the model --------------------------------------------------


def test_horizon_follows_the_threat_class():
    confidential = AssetModel("d", "X25519", data_lifetime_years=12)
    credential = AssetModel(
        "c", "ECDSA", data_lifetime_years=12,
        threat_class=ThreatClass.AUTHENTICATION, credential_validity_years=0.25,
    )
    artifact = AssetModel(
        "s", "ECDSA", data_lifetime_years=12,
        threat_class=ThreatClass.NON_REPUDIATION, verification_horizon_years=25,
    )
    assert horizon_for(confidential) == 12
    assert horizon_for(credential) == 0.25
    assert horizon_for(artifact) == 25


def test_calibrated_node_risk_separates_a_leaf_from_a_root():
    shared = dict(sensitivity=0.8, exposure=0.8, data_lifetime_years=15,
                  threat_class=ThreatClass.AUTHENTICATION)
    leaf = AssetModel("leaf", "ECDSA", credential_validity_years=0.25, **shared)
    root = AssetModel("root", "ECDSA", credential_validity_years=20, **shared)
    assert node_risk(root, calibration=DEFAULT_CALIBRATION) > 10 * node_risk(
        leaf, calibration=DEFAULT_CALIBRATION
    )


def test_calibration_is_opt_in_and_changes_the_score():
    asset = AssetModel("a", "AES-256", 0.9, 0.9, 20, rotation_hours=24)
    assert node_risk(asset) > 0                       # uncalibrated: nonzero
    assert node_risk(asset, calibration=DEFAULT_CALIBRATION) == 0.0  # calibrated


def test_calibrated_factor_stays_on_the_unit_scale():
    for algorithm in ("RSA-2048", "ECDSA", "AES-256", "ML-KEM", "unknown-thing"):
        for horizon in (0.25, 1, 5, 20, 100):
            value = calibrated_quantum_factor(algorithm, horizon)
            assert 0.0 <= value <= 1.0


def test_config_bundles_every_knob_and_reports_them():
    """Bundled so the set cannot drift apart between call sites, and serialised
    in full so a report records the assumptions behind its numbers."""
    config = CalibrationConfig(
        EXPERT_FORECASTS["aggressive"],
        doubling_time_years=1.0,
        pqc_residual_risk=0.02,
        composition_risk=0.03,
    )
    assert config.as_dict() == {
        "forecast": "aggressive",
        "doubling_time_years": 1.0,
        "pqc_residual_risk": 0.02,
        "composition_risk": 0.03,
    }
    assert config.probability("ML-KEM", 10) == 0.02


def test_provenance_separates_the_three_tiers():
    record = provenance()
    assert set(record) >= {
        "tier_a_published_estimates",
        "tier_b_stated_assumption",
        "tier_c_unobservable",
        "not_calibrated_against_outcomes",
    }
    # The central honesty claim must survive refactoring.
    assert "no outcome events to fit against" in record["not_calibrated_against_outcomes"]


def test_custom_forecast_can_be_supplied():
    """An organisation with its own threat intelligence should be able to use it."""
    mine = CRQCForecast("house-view", ((5, 0.5), (10, 0.9)), "internal estimate")
    assert capability_probability("RSA-2048", 5, forecast=mine) == pytest.approx(0.5)
