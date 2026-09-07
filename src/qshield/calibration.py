"""Grounding the model's parameters in published data, and saying where that stops.

Q-SHIELD has called its parameters "uncalibrated hypotheses" since 0.3. That was
honest but lazy: it lumped together three very different kinds of number, and by
treating them all as ungroundable it never checked whether the groundable ones
were even *right*. They were not. See :data:`QUANTUM_RESOURCES` below — the
literature disagrees with the shipped factors in two places, and in one of them
the shipped model has the sign backwards.

**Full calibration is impossible and nobody can do it.** Calibration in the
strict sense means fitting parameters to observed outcomes. No cryptographically
relevant quantum computer (CRQC) exists, nothing has ever been compromised via
Shor's algorithm, and so there are exactly zero outcome events to fit against.
Any tool claiming a calibrated quantum risk score is claiming data that does not
exist. What follows is therefore not a calibration of the model against reality;
it is a replacement of invented constants with published quantitative estimates,
which is the strongest grounding available and is still not the same thing.

Three tiers, kept separate on purpose:

**Tier A — grounded in published quantitative estimates.**
    Relative quantum work per primitive, from resource-estimate papers
    (:data:`QUANTUM_RESOURCES`), and the CRQC arrival distribution, from a
    seven-year expert elicitation survey (:data:`EXPERT_FORECASTS`). Both are
    citable numbers produced by other people, not by this repository.

**Tier B — a stated modelling assumption, swept rather than believed.**
    Converting "primitive X needs 2.6x fewer logical qubits than primitive Y"
    into "X falls N years earlier" needs a hardware-progress rate. That rate is
    not calibrated. It is a named parameter with a default, and
    ``qshield.experiments.calibration_sensitivity`` sweeps it.

**Tier C — irreducibly unobservable, unchanged.**
    Objective weights, per-asset sensitivity, exposure, business criticality and
    migration cost. No literature will ever supply these for your estate. They
    remain assumptions and are handled by ``qshield.experiments.robustness``.

The payoff is that Tier A collapses two arbitrary factors into one quantity with
a real meaning. The pre-0.7 model multiplied a quantum factor by a longevity
term, neither of which denoted anything; the calibrated model computes

    P(the capability to break this primitive exists before this asset's
      exposure horizon ends)

which is a probability about the world, and which can be argued with.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .algorithms import Family, family, hybrid_components, normalize
from .threat import ThreatClass

__all__ = [
    "CalibrationConfig",
    "CRQCForecast",
    "EXPERT_FORECASTS",
    "QUANTUM_RESOURCES",
    "QuantumResource",
    "capability_probability",
    "calibrated_quantum_factor",
    "mosca_verdict",
]

# The year the shipped forecasts are anchored to. Arrival probabilities are
# quoted as "within N years" of this date.
FORECAST_BASE_YEAR = 2026


# --------------------------------------------------------------------------
# Tier A(i): how much quantum work each primitive actually takes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantumResource:
    """Published cost of breaking one primitive on a fault-tolerant machine."""

    logical_qubits: float
    toffoli_gates: float
    classical_security_bits: int
    source: str
    note: str = ""


# Costs are quoted at the classical security level, which is the only fair way to
# compare them: RSA-3072 and ECC-256 both target ~128-bit classical security, so
# they are the like-for-like pair.
#
# The headline finding, and it contradicts every version of this model through
# 0.6: **elliptic curve cryptography is a substantially easier quantum target
# than RSA at equal classical security.** ECC-256 needs 2.6x fewer logical qubits
# and ~148x fewer Toffoli gates than RSA-3072. The shipped registry scores both
# at exactly 1.00, which asserts they fall at the same moment. They do not; ECC
# falls first, and an estate that migrated RSA before ECDSA on the strength of
# this model migrated them in the wrong order.
QUANTUM_RESOURCES: Mapping[str, QuantumResource] = {
    "RSA-3072": QuantumResource(
        logical_qubits=6146,
        toffoli_gates=1.86e13,
        classical_security_bits=128,
        source="Roetteler, Naehrig, Svore & Lauter, ASIACRYPT 2017 (arXiv:1706.06752)",
    ),
    "ECDSA": QuantumResource(
        logical_qubits=2330,
        toffoli_gates=1.26e11,
        classical_security_bits=128,
        source="Roetteler, Naehrig, Svore & Lauter, ASIACRYPT 2017 (arXiv:1706.06752)",
        note="NIST P-256 point-addition circuit; applies to any 256-bit prime-field curve",
    ),
    # RSA-2048 is the reference the expert survey is anchored on, so it is
    # assigned ratio 1.0 by construction rather than by an independent estimate.
    "RSA-2048": QuantumResource(
        logical_qubits=4098,
        toffoli_gates=5.5e12,
        classical_security_bits=112,
        source="Gidney & Ekerå, Quantum 5, 433 (2021) (arXiv:1905.09749); "
        "Gidney (arXiv:2505.15917) later reduces the physical-qubit requirement "
        "to under one million",
        note="the anchor primitive: the expert survey asks specifically about RSA-2048",
    ),
}

# Curves and key-exchange variants share the ECDLP cost at their security level.
_RESOURCE_ALIASES: Mapping[str, str] = {
    "ECDH": "ECDSA",
    "ECC": "ECDSA",
    "X25519": "ECDSA",
    "ED25519": "ECDSA",
    "ED448": "ECDSA",
    "DSA": "RSA-3072",   # finite-field DLP, comparable to factoring at equal size
    "DH": "RSA-3072",
    "RSA-1024": "RSA-2048",
    "RSA-4096": "RSA-3072",
}

# Symmetric primitives are not on this scale at all. Grover gives a quadratic
# speedup, and a quadratic speedup on a 128-bit key is not an attack: NIST makes
# AES-128 the *benchmark* against which post-quantum candidates are measured, and
# key search on AES-256 costs on the order of 2^298/MAXDEPTH quantum gates.
# Grover also parallelises poorly, so the depth restriction bites hard.
#
# Every version through 0.6 scored AES-128 at 0.15 and AES-256 at 0.08 -- that is
# 1.6x to 3x the 0.05 it assigned ML-KEM. The model was assigning symmetric
# cryptography *more* quantum risk than the post-quantum algorithms it recommends
# migrating to. That is backwards, and it inflated the baseline risk of every
# data-at-rest asset in every case file.
SYMMETRIC_CAPABILITY_PROBABILITY = 0.0
SYMMETRIC_SOURCE = (
    "NIST PQC security categories (AES-128/192/256 define levels 1/3/5); "
    "Jaques, Naehrig, Roetteler & Virdia, EUROCRYPT 2020 (arXiv:1910.01700)"
)

# Post-quantum primitives carry residual risk, but it is not quantum
# susceptibility -- it is the ordinary risk of a young primitive: cryptanalytic
# progress, implementation and side-channel defects, parameter mistakes. Keeping
# it in the same units as a Shor break was always a category error; it is kept as
# an explicit, separately-named floor so it can be argued with on its own terms.
PQC_RESIDUAL_RISK = 0.05
PQC_RESIDUAL_SOURCE = (
    "not a quantum-susceptibility estimate: a placeholder for cryptanalytic and "
    "implementation risk in a young primitive, carried over from earlier versions"
)


# --------------------------------------------------------------------------
# Tier A(ii): when the capability arrives
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CRQCForecast:
    """Cumulative probability that a CRQC able to break RSA-2048 exists.

    Anchors are (years from :data:`FORECAST_BASE_YEAR`, probability) pairs taken
    from published expert elicitation. Between anchors the curve is interpolated
    linearly in probability; beyond the last anchor it is held flat rather than
    extrapolated, because extrapolating an elicitation past the horizon the
    experts were asked about invents data.
    """

    name: str
    anchors: Sequence[tuple[float, float]]
    source: str

    def probability_within(self, years: float) -> float:
        if years <= 0:
            return 0.0
        anchors = sorted(self.anchors)
        if years <= anchors[0][0]:
            # Linear from (0, 0) to the first anchor.
            first_years, first_probability = anchors[0]
            return first_probability * (years / first_years)
        for (y0, p0), (y1, p1) in zip(anchors, anchors[1:], strict=False):
            if years <= y1:
                span = y1 - y0
                return p0 + (p1 - p0) * ((years - y0) / span) if span else p1
        return anchors[-1][1]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "anchors": [list(a) for a in self.anchors],
            "source": self.source,
            "base_year": FORECAST_BASE_YEAR,
        }


# The Global Risk Institute / evolutionQ "Quantum Threat Timeline" is the
# longest-running expert elicitation on this question. Its 2025 edition (26
# experts) reports the probability of a CRQC within 10 years as **28-49%** and
# within 15 years as **51-70%**, the range reflecting how responses are
# aggregated. Rather than collapse that to a midpoint and present it as precise,
# the two endpoints are carried as separate scenarios and the disagreement is
# propagated into the results.
_SURVEY_SOURCE = (
    "Mosca & Piani, Quantum Threat Timeline Report 2025, Global Risk Institute / "
    "evolutionQ (26 experts): CRQC within 10 years 28-49%, within 15 years 51-70%. "
    "Earlier editions place a break within 5 years at under 1%."
)

EXPERT_FORECASTS: Mapping[str, CRQCForecast] = {
    "conservative": CRQCForecast(
        name="conservative",
        anchors=((5, 0.01), (10, 0.28), (15, 0.51), (20, 0.70), (30, 0.85)),
        source=_SURVEY_SOURCE + " Low end of each published range.",
    ),
    "central": CRQCForecast(
        name="central",
        anchors=((5, 0.03), (10, 0.385), (15, 0.605), (20, 0.78), (30, 0.90)),
        source=_SURVEY_SOURCE + " Midpoint of each published range.",
    ),
    "aggressive": CRQCForecast(
        name="aggressive",
        anchors=((5, 0.05), (10, 0.49), (15, 0.70), (20, 0.86), (30, 0.95)),
        source=_SURVEY_SOURCE + " High end of each published range.",
    ),
}
DEFAULT_FORECAST = EXPERT_FORECASTS["central"]

# The 20- and 30-year anchors extend past the horizons the survey reports. They
# are a monotone continuation, not elicited values, and are flagged as such.
EXTRAPOLATED_ANCHOR_YEARS = (20, 30)


# --------------------------------------------------------------------------
# Tier B: turning a resource ratio into a time offset
# --------------------------------------------------------------------------

# How long it takes available logical qubit counts to double. This is the one
# free parameter of the calibration and it is NOT calibrated: it is a stated
# assumption, defaulted from the broad shape of published hardware roadmaps and
# swept in qshield.experiments.calibration_sensitivity. A primitive needing
# fewer logical qubits than RSA-2048 becomes reachable earlier, by
# doubling_time_years * log2(ratio).
DEFAULT_QUBIT_DOUBLING_YEARS = 2.0


def resource_for(algorithm: str) -> QuantumResource | None:
    key = normalize(algorithm)
    key = _RESOURCE_ALIASES.get(key, key)
    return QUANTUM_RESOURCES.get(key)


def lead_time_years(
    algorithm: str, *, doubling_time_years: float = DEFAULT_QUBIT_DOUBLING_YEARS
) -> float:
    """How much earlier this primitive falls than RSA-2048, in years.

    Positive means earlier. Derived from the ratio of logical qubit requirements
    under an exponential hardware-growth assumption; the assumption is Tier B and
    the ratio is Tier A.
    """
    resource = resource_for(algorithm)
    anchor = QUANTUM_RESOURCES["RSA-2048"]
    if resource is None or doubling_time_years <= 0:
        return 0.0
    ratio = anchor.logical_qubits / resource.logical_qubits
    return doubling_time_years * math.log2(ratio)


def capability_probability(
    algorithm: str,
    horizon_years: float,
    *,
    forecast: CRQCForecast = DEFAULT_FORECAST,
    doubling_time_years: float = DEFAULT_QUBIT_DOUBLING_YEARS,
) -> float:
    """P(the capability to break ``algorithm`` exists within ``horizon_years``).

    This is the quantity that replaces the old ``quantum_factor * longevity``
    product. Both of those were dimensionless inventions; this is a probability,
    anchored on a published expert survey and shifted per primitive by published
    resource estimates.
    """
    if horizon_years <= 0:
        return 0.0
    fam = family(algorithm)
    if fam is Family.POST_QUANTUM:
        return PQC_RESIDUAL_RISK
    if fam is Family.GROVER_WEAKENED:
        return SYMMETRIC_CAPABILITY_PROBABILITY
    if fam is not Family.SHOR_VULNERABLE:
        # Unknown primitive: the model cannot say, and guessing low would be the
        # dangerous direction to guess in.
        return forecast.probability_within(horizon_years)
    effective = horizon_years + lead_time_years(
        algorithm, doubling_time_years=doubling_time_years
    )
    return forecast.probability_within(effective)


def calibrated_quantum_factor(
    algorithm: str,
    horizon_years: float,
    *,
    forecast: CRQCForecast = DEFAULT_FORECAST,
    doubling_time_years: float = DEFAULT_QUBIT_DOUBLING_YEARS,
) -> float:
    """The calibrated replacement for ``quantum_factor(alg) * longevity(horizon)``.

    Returned on the same 0-1 scale the objective already expects, so it drops
    into :func:`qshield.model.node_risk` without rescaling anything else.
    """
    return capability_probability(
        algorithm,
        horizon_years,
        forecast=forecast,
        doubling_time_years=doubling_time_years,
    )


@dataclass(frozen=True)
class CalibrationConfig:
    """Everything the calibrated scoring path needs, in one passable object.

    Bundled rather than threaded as loose arguments because the set of functions
    needing them is exactly the set that already takes the objective weights, and
    passing three parameters separately invites supplying them at some call sites
    and not others -- the bug that let 0.3's two optimizers drift apart.

    ``pqc_residual_risk`` is Tier C, not Tier A: it is the risk of a young
    primitive, not a quantum-susceptibility estimate, and it decides the crossover
    horizon below which migrating is modelled as net-negative. Sweep it before
    relying on any conclusion that turns on it.
    """

    forecast: CRQCForecast = DEFAULT_FORECAST
    doubling_time_years: float = DEFAULT_QUBIT_DOUBLING_YEARS
    pqc_residual_risk: float = PQC_RESIDUAL_RISK
    # Risk that a hybrid composition fails even though neither half is broken:
    # a bad combiner, a downgrade path, one half silently not negotiated. Tier C,
    # and the floor on how safe a hybrid can be.
    composition_risk: float = 0.01

    def probability(self, algorithm: str, horizon_years: float) -> float:
        fam = family(algorithm)
        if fam is Family.POST_QUANTUM:
            return self.pqc_residual_risk
        if fam is Family.HYBRID:
            return self._hybrid_probability(algorithm, horizon_years)
        return capability_probability(
            algorithm,
            horizon_years,
            forecast=self.forecast,
            doubling_time_years=self.doubling_time_years,
        )

    def _hybrid_probability(self, algorithm: str, horizon_years: float) -> float:
        """A hybrid is broken only if both halves are.

        The product treats the two breaks as independent, which is the honest
        reading: a CRQC that solves elliptic-curve discrete log gives no purchase
        on a lattice problem, and lattice cryptanalysis gives none on discrete
        log. Correlated *implementation* failure is real, and it is priced
        explicitly as ``composition_risk`` rather than left out.
        """
        parts = hybrid_components(algorithm)
        if parts is None:  # pragma: no cover - family() already established this
            return capability_probability(algorithm, horizon_years)
        classical, post_quantum = parts
        both = self.probability(classical, horizon_years) * self.probability(
            post_quantum, horizon_years
        )
        combined = both + self.composition_risk - both * self.composition_risk
        return max(self.composition_risk, min(1.0, combined))

    def as_dict(self) -> dict[str, object]:
        return {
            "forecast": self.forecast.name,
            "doubling_time_years": self.doubling_time_years,
            "pqc_residual_risk": self.pqc_residual_risk,
            "composition_risk": self.composition_risk,
        }


DEFAULT_CALIBRATION = CalibrationConfig()


def migration_is_net_positive(
    algorithm: str,
    replacement: str,
    horizon_years: float,
    *,
    config: CalibrationConfig = DEFAULT_CALIBRATION,
) -> bool:
    """Whether replacing ``algorithm`` with ``replacement`` lowers modelled risk.

    Not always true under calibration, and that is the point. Over a short enough
    horizon the probability that a CRQC arrives at all is smaller than the
    residual risk of a young post-quantum primitive, so the swap is modelled as
    net-negative. The uncalibrated model could not express this: it multiplied a
    quantum factor of 1.00 by a longevity term, so migration always helped by
    construction.
    """
    return config.probability(replacement, horizon_years) < config.probability(
        algorithm, horizon_years
    )


def crossover_horizon_years(
    algorithm: str,
    replacement: str,
    *,
    config: CalibrationConfig = DEFAULT_CALIBRATION,
    limit: float = 40.0,
    step: float = 0.05,
) -> float:
    """Shortest horizon at which migrating ``algorithm`` starts to pay.

    Returns 0.0 when migration always pays and ``inf`` when it never does within
    ``limit``.
    """
    horizon = 0.0
    while horizon <= limit:
        if migration_is_net_positive(algorithm, replacement, horizon, config=config):
            return horizon
        horizon += step
    return math.inf


# --------------------------------------------------------------------------
# Mosca's inequality
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MoscaVerdict:
    """Whether migration started late, by how much, and on what assumptions."""

    shelf_life_years: float
    migration_years: float
    years_until_capability: float
    deadline_missed: bool
    slack_years: float
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "shelf_life_years": round(self.shelf_life_years, 3),
            "migration_years": round(self.migration_years, 3),
            "years_until_capability": round(self.years_until_capability, 3),
            "deadline_missed": self.deadline_missed,
            "slack_years": round(self.slack_years, 3),
            "at_confidence": self.confidence,
            "rule": "x (shelf life) + y (migration time) > z (time to capability)",
        }


def mosca_verdict(
    *,
    shelf_life_years: float,
    migration_years: float,
    algorithm: str = "RSA-2048",
    forecast: CRQCForecast = DEFAULT_FORECAST,
    doubling_time_years: float = DEFAULT_QUBIT_DOUBLING_YEARS,
    confidence: float = 0.5,
) -> MoscaVerdict:
    """Mosca's inequality: if x + y > z, the migration is already late.

    ``z`` is taken as the year by which the forecast reaches ``confidence``, so
    the verdict is explicit about the risk appetite it encodes rather than
    hiding one. This has no free parameters of its own — it is arithmetic over
    the forecast — which is why it is the most defensible output here.
    """
    z = years_to_confidence(
        algorithm,
        confidence,
        forecast=forecast,
        doubling_time_years=doubling_time_years,
    )
    total = shelf_life_years + migration_years
    return MoscaVerdict(
        shelf_life_years=shelf_life_years,
        migration_years=migration_years,
        years_until_capability=z,
        deadline_missed=total > z,
        slack_years=z - total,
        confidence=confidence,
    )


def years_to_confidence(
    algorithm: str,
    confidence: float,
    *,
    forecast: CRQCForecast = DEFAULT_FORECAST,
    doubling_time_years: float = DEFAULT_QUBIT_DOUBLING_YEARS,
    horizon: float = 30.0,
) -> float:
    """First year at which the capability probability reaches ``confidence``.

    Returns ``inf`` when the forecast never reaches it inside ``horizon`` — which
    is the correct answer for symmetric primitives, and is why it is returned
    rather than clamped.
    """
    step = 0.25
    years = 0.0
    while years <= horizon:
        if (
            capability_probability(
                algorithm,
                years,
                forecast=forecast,
                doubling_time_years=doubling_time_years,
            )
            >= confidence
        ):
            return years
        years += step
    return math.inf


def horizon_for(asset) -> float:
    """The exposure horizon the calibrated probability should be measured over.

    Which clock matters depends on the threat class, exactly as in
    :mod:`qshield.threat`: harvested data is exposed for as long as it must stay
    secret, a credential for as long as it stays in service, a signed artifact
    for as long as it must remain verifiable.
    """
    cls = asset.effective_threat_class
    if cls is ThreatClass.AUTHENTICATION and asset.credential_validity_years is not None:
        return asset.credential_validity_years
    if (
        cls is ThreatClass.NON_REPUDIATION
        and asset.verification_horizon_years is not None
    ):
        return asset.verification_horizon_years
    return asset.data_lifetime_years


def provenance() -> dict[str, object]:
    """Every number in this module, with its source and its tier."""
    return {
        "tier_a_published_estimates": {
            "quantum_resources": {
                name: {
                    "logical_qubits": r.logical_qubits,
                    "toffoli_gates": r.toffoli_gates,
                    "classical_security_bits": r.classical_security_bits,
                    "source": r.source,
                    "note": r.note,
                }
                for name, r in QUANTUM_RESOURCES.items()
            },
            "symmetric": {
                "capability_probability": SYMMETRIC_CAPABILITY_PROBABILITY,
                "source": SYMMETRIC_SOURCE,
            },
            "crqc_forecasts": {
                name: f.as_dict() for name, f in EXPERT_FORECASTS.items()
            },
            "extrapolated_anchor_years": list(EXTRAPOLATED_ANCHOR_YEARS),
        },
        "tier_b_stated_assumption": {
            "qubit_doubling_years": DEFAULT_QUBIT_DOUBLING_YEARS,
            "why": (
                "converts a published resource ratio into a time offset; not "
                "calibrated, swept in qshield.experiments.calibration_sensitivity"
            ),
        },
        "tier_c_unobservable": {
            "fields": [
                "objective weights",
                "sensitivity",
                "exposure",
                "business_criticality",
                "migration_cost",
            ],
            "why": (
                "no literature supplies these for a specific estate; handled by "
                "qshield.experiments.robustness, not by calibration"
            ),
            "pqc_residual_risk": {
                "value": PQC_RESIDUAL_RISK,
                "source": PQC_RESIDUAL_SOURCE,
            },
        },
        "not_calibrated_against_outcomes": (
            "No CRQC exists and nothing has been compromised via Shor's algorithm, "
            "so there are no outcome events to fit against. These are published "
            "estimates substituted for invented constants, which is not the same "
            "as a model validated against reality."
        ),
    }
