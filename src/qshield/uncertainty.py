"""Uncertainty propagation and paired comparison.

0.3 ran an independent Monte Carlo per strategy with a *different seed for each*
(11 for the baseline, 12 for flat, 13 for the optimizer) and reported three sets
of percentiles side by side. Those intervals are not comparable: each strategy
was scored on a different random world, so the spread between them mixes the
strategy effect with sampling noise, and there is no way to recover the
difference that actually matters.

This module uses **common random numbers**. One perturbation is drawn per trial
and applied to every strategy, so the difference between two strategies is
measured on identical inputs. The variance of the paired difference is far lower
than the variance of either arm — which is why 0.3's overlapping percentile
bands were uninformative — and it supports a bootstrap confidence interval on
the quantity the research question is about.

The perturbation model also covers what 0.3 left fixed. 0.3 sampled sensitivity,
exposure, lifetime and edge reliability, but held the quantum factors and the
objective weights constant even though its own README calls those "hypotheses".
Uncertainty in the parameters you are least sure about was therefore excluded
from the reported intervals. :class:`PerturbationModel` can sample both.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from .algorithms import quantum_factor
from .model import AssetModel, ObjectiveWeights
from .paths import EdgeModel

__all__ = [
    "PerturbationModel",
    "World",
    "sample_worlds",
    "quantiles",
    "summarize",
    "paired_comparison",
    "bootstrap_ci",
]


@dataclass(frozen=True)
class PerturbationModel:
    """How far each uncertain quantity is allowed to move.

    Defaults reproduce 0.3's spreads for the parameters it did sample, so results
    remain comparable; ``quantum_factor_spread`` and ``weight_spread`` default to
    off and must be switched on deliberately, since they change what the reported
    interval means.
    """

    unit_spread: float = 0.08          # triangular half-width on sensitivity/exposure
    lifetime_spread: float = 0.15      # multiplicative, uniform
    reliability_low: float = 0.90      # multiplicative, uniform
    reliability_high: float = 1.05
    quantum_factor_spread: float = 0.0  # multiplicative, uniform; 0 disables
    weight_spread: float = 0.0          # multiplicative on each weight; 0 disables


@dataclass(frozen=True)
class World:
    """One sampled realisation of the uncertain inputs.

    Holds perturbed *inputs only* — never a plan — so the same world can be
    replayed against every strategy under comparison.

    Quantum-factor uncertainty is stored as a per-asset *multiplier*, not as an
    absolute override. A migration changes the asset's algorithm, so an absolute
    override sampled from the pre-migration algorithm would follow the asset
    across the migration and silently cancel it out. :meth:`overrides_for`
    resolves the multiplier against whatever algorithm the asset holds in the
    plan being scored.
    """

    assets: tuple[AssetModel, ...]
    edges: tuple[EdgeModel, ...]
    qf_multipliers: Mapping[str, float]
    weights: ObjectiveWeights

    def overrides_for(self, assets: Sequence[AssetModel]) -> dict[str, float]:
        """Absolute quantum factors for ``assets`` under this world's draw."""
        if not self.qf_multipliers:
            return {}
        return {
            a.name: _clamp(quantum_factor(a.algorithm) * self.qf_multipliers[a.name], 1e-6, 1.0)
            for a in assets
            if a.name in self.qf_multipliers
        }


def sample_worlds(
    assets: Sequence[AssetModel],
    edges: Sequence[EdgeModel],
    *,
    trials: int,
    seed: int,
    weights: ObjectiveWeights,
    model: PerturbationModel = PerturbationModel(),
) -> list[World]:
    """Draw ``trials`` worlds. Deterministic given ``seed``.

    Every strategy in a comparison must be scored against this same list — that
    is the whole point of common random numbers.
    """
    rng = random.Random(seed)
    worlds: list[World] = []
    for _ in range(trials):
        sampled: list[AssetModel] = []
        overrides: dict[str, float] = {}
        for a in assets:
            sampled.append(
                replace(
                    a,
                    sensitivity=_triangular(rng, a.sensitivity, model.unit_spread),
                    exposure=_triangular(rng, a.exposure, model.unit_spread),
                    data_lifetime_years=max(
                        0.1,
                        a.data_lifetime_years
                        * rng.uniform(1 - model.lifetime_spread, 1 + model.lifetime_spread),
                    ),
                )
            )
            if model.quantum_factor_spread > 0:
                overrides[a.name] = rng.uniform(
                    1 - model.quantum_factor_spread, 1 + model.quantum_factor_spread
                )
        perturbed_edges = tuple(
            replace(
                e,
                reliability=_clamp(
                    e.reliability * rng.uniform(model.reliability_low, model.reliability_high)
                ),
            )
            for e in edges
        )
        w = weights
        if model.weight_spread > 0:
            lo, hi = 1 - model.weight_spread, 1 + model.weight_spread
            w = ObjectiveWeights(
                max(1e-9, weights.node * rng.uniform(lo, hi)),
                max(0.0, weights.path * rng.uniform(lo, hi)),
                max(0.0, weights.rotation * rng.uniform(lo, hi)),
            )
        worlds.append(World(tuple(sampled), perturbed_edges, overrides, w))
    return worlds


def quantiles(values: Sequence[float]) -> dict[str, float]:
    """Percentiles by linear interpolation.

    0.3 indexed the sorted list directly and inconsistently — ``p05`` used
    ``int(.05*n)-1`` while ``p95`` used ``int(.95*n)``, and the "median" took the
    upper element for even ``n``. The discrepancy is small but it biased every
    reported interval in one direction.
    """
    if not values:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "p05": _quantile(ordered, 0.05),
        "p50": _quantile(ordered, 0.50),
        "p95": _quantile(ordered, 0.95),
    }


def summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "stdev": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    out: dict[str, float] = {"n": len(values), "mean": statistics.fmean(values)}
    out["stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    out.update(quantiles(values))
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260907,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic``.

    Used instead of a t-interval because paired objective differences are heavily
    zero-inflated — most instances yield identical plans — and badly non-normal.
    """
    if len(values) < 2:
        v = float(values[0]) if values else 0.0
        return (v, v)
    rng = random.Random(seed)
    n = len(values)
    stats: list[float] = []
    for _ in range(resamples):
        stats.append(statistic([values[rng.randrange(n)] for _ in range(n)]))
    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    return (_quantile(stats, alpha), _quantile(stats, 1.0 - alpha))


def paired_comparison(
    deltas: Sequence[float],
    *,
    tolerance: float = 1e-9,
    label_a: str = "a",
    label_b: str = "b",
    seed: int = 20260907,
) -> dict[str, object]:
    """Summarise paired differences ``value(a) - value(b)`` over instances.

    Positive delta means ``b`` scored lower (better). Reports win/loss/tie counts,
    a bootstrap CI on the mean, and — the number 0.3 never gave — the effect
    conditional on the plans actually differing. When most instances tie, an
    unconditional mean understates the effect where it exists and a raw win rate
    overstates how often the method matters.
    """
    deltas = list(deltas)
    n = len(deltas)
    if n == 0:
        return {"instances": 0}
    wins = [d for d in deltas if d > tolerance]      # b better
    losses = [d for d in deltas if d < -tolerance]   # a better
    ties = n - len(wins) - len(losses)
    decisive = wins + losses
    lo, hi = bootstrap_ci(deltas, seed=seed)
    out: dict[str, object] = {
        "comparison": f"{label_a} - {label_b}",
        "instances": n,
        f"{label_b}_better": len(wins),
        f"{label_a}_better": len(losses),
        "ties": ties,
        "tie_rate": round(ties / n, 6),
        "mean_delta": round(statistics.fmean(deltas), 6),
        "mean_delta_ci95": [round(lo, 6), round(hi, 6)],
        "median_delta": round(statistics.median(deltas), 6),
        "max_delta": round(max(deltas), 6),
        "min_delta": round(min(deltas), 6),
        # Excludes ties: "when the two strategies disagree, how much does it cost?"
        "decisive_instances": len(decisive),
        "mean_delta_when_decisive": round(statistics.fmean(decisive), 6) if decisive else 0.0,
    }
    if decisive:
        d_lo, d_hi = bootstrap_ci(decisive, seed=seed + 1)
        out["mean_delta_when_decisive_ci95"] = [round(d_lo, 6), round(d_hi, 6)]
        out["win_rate_given_decisive"] = round(len(wins) / len(decisive), 6)
    return out


def _triangular(rng: random.Random, mode: float, spread: float) -> float:
    low = max(0.0, mode - spread)
    high = min(1.0, mode + spread)
    if high <= low:
        return _clamp(mode)
    return _clamp(rng.triangular(low, high, min(max(mode, low), high)))


def _quantile(ordered: Sequence[float], q: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
