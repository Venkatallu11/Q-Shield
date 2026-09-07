"""Does grounding the parameters in published data change what you should do?

Replacing invented constants with published estimates is defensible on its own
terms, but "more principled" is not the same as "changes the answer". If the
recommendation is identical either way, the calibration is bookkeeping.

It is not framed as a win rate, and cannot be. The two models optimise different
objectives, and there is no ground truth to say which is right — no CRQC exists,
so no outcome data can adjudicate between them. What is measured is the *size of
the disagreement*, which was free to come out zero.

Three things move between the models, and they push in different directions:

* **ECC is a much easier quantum target than RSA at equal classical security**
  (2.6x fewer logical qubits, ~148x fewer Toffoli gates). The uncalibrated
  registry scored both at exactly 1.00, asserting they fall at the same moment.
  Calibration pulls ECDSA/X25519/Ed25519 *up* relative to RSA.
* **Symmetric primitives are not meaningfully threatened.** NIST makes AES-128
  the benchmark for post-quantum security. The uncalibrated registry scored
  AES-128 at 0.15 and AES-256 at 0.08 — above the 0.05 it gave ML-KEM. Calibration
  pushes them to zero, deflating the baseline risk of every data-at-rest asset.
* **Short horizons stop justifying migration.** The calibrated term is a
  probability that a capability arrives before the asset's horizon ends. Below a
  crossover horizon that probability is smaller than the residual risk of a young
  post-quantum primitive, so the swap is modelled as net-negative. The
  uncalibrated model multiplied a quantum factor of 1.00 by a longevity term and
  so could never express this.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

from ..calibration import (
    DEFAULT_CALIBRATION,
    EXPERT_FORECASTS,
    CalibrationConfig,
    crossover_horizon_years,
    horizon_for,
)
from ..model import DEFAULT_WEIGHTS, ObjectiveWeights
from ..optimizer import MigrationProblem
from ..planner import plan as choose_plan
from ..uncertainty import summarize
from .generator import make_instances as generic_instances
from .pki import PKIConfig
from .pki import make_instances as pki_instances


def calibrated(problem: MigrationProblem, config: CalibrationConfig) -> MigrationProblem:
    return replace(problem, calibration=config)


def run(
    *,
    instances: int = 300,
    seed: int = 20260907,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    config: CalibrationConfig = DEFAULT_CALIBRATION,
    pki_config: PKIConfig = PKIConfig(),
) -> dict[str, object]:
    suites = {
        "pki": pki_instances(instances, seed, pki_config),
        "generic": generic_instances(instances, seed),
    }

    results: dict[str, object] = {}
    for suite, problems in suites.items():
        problems = [p for p in problems if p.candidates]
        agree = 0
        both_empty = 0
        calibrated_empty = 0
        uncalibrated_empty = 0
        spend_delta: list[float] = []
        gained: Counter[str] = Counter()
        dropped: Counter[str] = Counter()

        for problem in problems:
            plain = set(choose_plan(problem, weights=weights).selected)
            cal = set(choose_plan(calibrated(problem, config), weights=weights).selected)
            if plain == cal:
                agree += 1
            if not plain and not cal:
                both_empty += 1
            if not cal:
                calibrated_empty += 1
            if not plain:
                uncalibrated_empty += 1
            spend_delta.append(problem.cost_of(cal) - problem.cost_of(plain))
            by_name = {a.name: a for a in problem.assets}
            for name in cal - plain:
                gained[_bucket(by_name[name].algorithm)] += 1
            for name in plain - cal:
                dropped[_bucket(by_name[name].algorithm)] += 1

        results[suite] = {
            "instances": len(problems),
            "plan_agreement": round(agree / len(problems), 6) if problems else 0.0,
            "calibrated_recommends_nothing": calibrated_empty,
            "uncalibrated_recommends_nothing": uncalibrated_empty,
            "both_recommend_nothing": both_empty,
            "spend_delta": summarize(spend_delta),
            "assets_gained_under_calibration": dict(gained.most_common()),
            "assets_dropped_under_calibration": dict(dropped.most_common()),
        }

    return {
        "experiment": "calibration_impact",
        "question": (
            "Does replacing the invented quantum factors and longevity term with "
            "published resource estimates and expert-elicited arrival "
            "probabilities change which assets get migrated?"
        ),
        "design": {
            "not_a_win_rate": (
                "the two models optimise different objectives and no outcome data "
                "exists to adjudicate between them; this measures the size of the "
                "disagreement, which was free to be zero"
            ),
            "calibration": config.as_dict(),
            "weights": weights.as_tuple(),
            "seed": seed,
        },
        "suites": results,
        "migration_crossover_years": _crossovers(config),
        "reading": (
            "A low agreement rate means the uncalibrated model was recommending "
            "different work, not merely scoring the same work differently."
        ),
    }


def _bucket(algorithm: str) -> str:
    from ..algorithms import Family, family, normalize

    key = normalize(algorithm)
    fam = family(key)
    if fam is Family.GROVER_WEAKENED:
        return "symmetric"
    if fam is Family.POST_QUANTUM:
        return "post-quantum"
    if key.startswith("RSA") or key in {"DH", "DSA"}:
        return "rsa/finite-field"
    return "elliptic-curve"


def _crossovers(config: CalibrationConfig) -> dict[str, object]:
    """Below these horizons, migrating is modelled as raising risk."""
    pairs = (
        ("RSA-2048", "ML-KEM"),
        ("RSA-3072", "ML-KEM"),
        ("ECDSA", "ML-DSA"),
        ("X25519", "ML-KEM"),
        ("ED25519", "ML-DSA"),
    )
    out = {}
    for source, target in pairs:
        years = crossover_horizon_years(source, target, config=config)
        out[f"{source} -> {target}"] = None if years == float("inf") else round(years, 2)
    out["_note"] = (
        "the existence of a crossover follows from the calibrated form; its "
        "location depends on pqc_residual_risk, which is Tier C and uncalibrated"
    )
    return out


def sweep(
    *,
    instances: int = 200,
    seed: int = 20260907,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    doubling_times: Sequence[float] = (1.0, 2.0, 4.0),
    residuals: Sequence[float] = (0.01, 0.02, 0.05, 0.10),
    pki_config: PKIConfig = PKIConfig(),
) -> dict[str, object]:
    """How much of the calibrated answer rests on the parameters still assumed?

    Tier B is the hardware doubling time that converts a resource ratio into a
    time offset; Tier C here is the residual risk of a young post-quantum
    primitive, which sets the crossover horizon. Neither is calibrated, so the
    honest question is how far the recommendation moves across them.
    """
    problems = [p for p in pki_instances(instances, seed, pki_config) if p.candidates]
    baseline = [
        frozenset(choose_plan(calibrated(p, DEFAULT_CALIBRATION), weights=weights).selected)
        for p in problems
    ]

    rows: list[dict[str, object]] = []
    for forecast_name, forecast in EXPERT_FORECASTS.items():
        for doubling in doubling_times:
            for residual in residuals:
                config = CalibrationConfig(forecast, doubling, residual)
                plans = [
                    frozenset(
                        choose_plan(calibrated(p, config), weights=weights).selected
                    )
                    for p in problems
                ]
                same = sum(1 for a, b in zip(plans, baseline, strict=True) if a == b)
                rows.append({
                    "forecast": forecast_name,
                    "doubling_time_years": doubling,
                    "pqc_residual_risk": residual,
                    "agreement_with_default_calibration": round(same / len(plans), 6),
                    "recommends_nothing": sum(1 for p in plans if not p),
                    "mean_plan_size": round(
                        statistics.fmean([len(p) for p in plans]), 4
                    ),
                    "crossover_ecdsa_years": _crossover_value(config, "ECDSA", "ML-DSA"),
                })

    agreements = [float(r["agreement_with_default_calibration"]) for r in rows]
    return {
        "experiment": "calibration_sensitivity",
        "question": (
            "How much does the calibrated recommendation depend on the parameters "
            "the calibration could not ground?"
        ),
        "design": {
            "instances": len(problems),
            "settings_evaluated": len(rows),
            "tier_b_swept": "doubling_time_years, forecast scenario",
            "tier_c_swept": "pqc_residual_risk",
            "baseline": DEFAULT_CALIBRATION.as_dict(),
            "seed": seed,
        },
        "agreement_across_settings": summarize(agreements),
        "worst_case_agreement": round(min(agreements), 6) if agreements else 0.0,
        "settings": rows,
    }


def _crossover_value(config: CalibrationConfig, source: str, target: str):
    years = crossover_horizon_years(source, target, config=config)
    return None if years == float("inf") else round(years, 2)


def horizon_summary(problem: MigrationProblem) -> dict[str, object]:
    """Exposure horizons in an instance, and how they sit against the crossover.

    Useful on a real estate: it says how much of the inventory is short-lived
    enough that the calibrated model would leave it alone.
    """
    horizons = [(a.name, horizon_for(a)) for a in problem.candidates]
    crossover = crossover_horizon_years("ECDSA", "ML-DSA")
    below = [n for n, h in horizons if h < crossover]
    return {
        "candidates": len(horizons),
        "crossover_years": round(crossover, 2) if crossover != float("inf") else None,
        "below_crossover": below,
        "share_below_crossover": (
            round(len(below) / len(horizons), 4) if horizons else 0.0
        ),
    }
