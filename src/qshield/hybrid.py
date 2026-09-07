"""Hybrid deployments: classical and post-quantum together.

Real migrations rarely jump from RSA to ML-KEM. They go through a hybrid, and
that is not a transitional inconvenience — it is what is actually deployed today.
X25519MLKEM768 is the default key exchange in current browsers. A migration
planner with no concept of hybrid is not modelling the migration that is
happening.

The reason hybrid matters here is sharper than "it is a middle step". A hybrid is
broken only if **both** halves are broken, so it covers the two failure modes
that a single primitive cannot cover at once:

* a CRQC breaks the classical half, and the post-quantum half holds;
* cryptanalysis of a young post-quantum primitive breaks that half, and the
  classical half holds.

Under the 0.7 calibration this has a consequence nobody has to take on faith.
Pure post-quantum carries a residual risk that is *not* quantum susceptibility —
it is the ordinary risk of a young primitive — and that residual is what creates
the crossover horizon below which migrating is modelled as making things worse.
A hybrid multiplies that residual by the probability the classical half also
falls, which is small at short horizons. So hybrid should pay off at horizons
where pure post-quantum does not. :mod:`qshield.experiments.hybrid_ladder`
measures whether it does.

Composition is not free and is not modelled as free. Concatenating two key
exchanges means a larger handshake, two implementations to get right, and a
combiner that can itself be wrong. That is carried as an explicit
``composition_risk`` floor rather than being quietly omitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .algorithms import Family, Primitive, family, normalize, primitive

__all__ = [
    "Deployment",
    "HYBRID_SUITES",
    "HybridSuite",
    "classify_deployment",
    "hybrid_probability",
    "migration_ladder",
    "parse_hybrid",
]

# Risk that the *composition* fails even though neither primitive is broken: a
# bad combiner, a downgrade path, one half silently not negotiated. Tier C -- an
# assumption, not a measurement -- and it is the floor on how safe a hybrid can
# be, so it is worth sweeping before relying on a hybrid recommendation.
DEFAULT_COMPOSITION_RISK = 0.01

# The separator used in registry names for hybrids.
SEPARATOR = "+"


class Deployment(str, Enum):
    """What a deployed asset actually runs, as opposed to which algorithm it names."""

    CLASSICAL = "classical"
    HYBRID = "hybrid"
    POST_QUANTUM = "post-quantum"
    SYMMETRIC = "symmetric"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HybridSuite:
    """A named composition of one classical and one post-quantum primitive."""

    name: str
    classical: str
    post_quantum: str
    primitive: Primitive
    deployed_as: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classical": self.classical,
            "post_quantum": self.post_quantum,
            "primitive": self.primitive.value,
            "deployed_as": self.deployed_as,
        }


# Compositions that are actually deployed or standardised, rather than every
# combination that could be written down.
HYBRID_SUITES: Mapping[str, HybridSuite] = {
    "X25519+ML-KEM": HybridSuite(
        "X25519+ML-KEM", "X25519", "ML-KEM", Primitive.KEY_ESTABLISHMENT,
        deployed_as="X25519MLKEM768, the default TLS key exchange in current browsers",
    ),
    "ECDH+ML-KEM": HybridSuite(
        "ECDH+ML-KEM", "ECDH", "ML-KEM", Primitive.KEY_ESTABLISHMENT,
        deployed_as="secp256r1MLKEM768",
    ),
    "RSA-2048+ML-KEM": HybridSuite(
        "RSA-2048+ML-KEM", "RSA-2048", "ML-KEM", Primitive.KEY_ESTABLISHMENT,
    ),
    "ECDSA+ML-DSA": HybridSuite(
        "ECDSA+ML-DSA", "ECDSA", "ML-DSA", Primitive.SIGNATURE,
        deployed_as="composite and chameleon certificate drafts",
    ),
    "ED25519+ML-DSA": HybridSuite(
        "ED25519+ML-DSA", "ED25519", "ML-DSA", Primitive.SIGNATURE,
    ),
    "RSA-2048+ML-DSA": HybridSuite(
        "RSA-2048+ML-DSA", "RSA-2048", "ML-DSA", Primitive.SIGNATURE,
    ),
}


def parse_hybrid(algorithm: str) -> HybridSuite | None:
    """Recognise a hybrid, whether registered by name or written as ``A+B``."""
    key = normalize(algorithm)
    suite = HYBRID_SUITES.get(key)
    if suite is not None:
        return suite
    if SEPARATOR not in key:
        return None
    left, _, right = key.partition(SEPARATOR)
    left, right = normalize(left), normalize(right)
    classical, post_quantum = None, None
    for part in (left, right):
        if family(part) is Family.POST_QUANTUM:
            post_quantum = part
        elif family(part) is Family.SHOR_VULNERABLE:
            classical = part
    if classical is None or post_quantum is None:
        return None
    return HybridSuite(
        f"{classical}{SEPARATOR}{post_quantum}",
        classical,
        post_quantum,
        primitive(post_quantum),
    )


def classify_deployment(algorithm: str) -> Deployment:
    if parse_hybrid(algorithm) is not None:
        return Deployment.HYBRID
    fam = family(algorithm)
    if fam is Family.SHOR_VULNERABLE:
        return Deployment.CLASSICAL
    if fam is Family.POST_QUANTUM:
        return Deployment.POST_QUANTUM
    if fam is Family.GROVER_WEAKENED:
        return Deployment.SYMMETRIC
    return Deployment.UNKNOWN


def hybrid_probability(
    suite: HybridSuite,
    classical_probability: float,
    post_quantum_probability: float,
    *,
    composition_risk: float = DEFAULT_COMPOSITION_RISK,
) -> float:
    """P(a hybrid is broken) = P(both halves broken) + P(the composition fails).

    The product treats the two breaks as independent, which is the honest
    reading: a CRQC that solves elliptic-curve discrete log gives no purchase on
    a lattice problem, and lattice cryptanalysis gives none on discrete log.
    Shared *implementation* failure is the correlated part, and that is exactly
    what ``composition_risk`` carries — so the assumption is not being hidden,
    it is being priced.

    Bounded below by ``composition_risk`` and above by the weaker half, since a
    hybrid can never be worse than its stronger component plus the cost of
    gluing them together.
    """
    both = classical_probability * post_quantum_probability
    combined = both + composition_risk - both * composition_risk
    return max(composition_risk, min(1.0, combined))


def migration_ladder(algorithm: str) -> tuple[str, ...]:
    """The deployment steps available from here, weakest first.

    A planner that can only offer ``classical -> post-quantum`` cannot express
    the migration most estates are actually performing. Returning the ladder lets
    the optimizer choose the rung rather than assuming the top one.
    """
    key = normalize(algorithm)
    deployment = classify_deployment(key)
    if deployment is Deployment.CLASSICAL:
        hybrid = default_hybrid_for(key)
        pure = _pure_replacement(key)
        rungs = [r for r in (hybrid, pure) if r]
        return tuple(rungs)
    if deployment is Deployment.HYBRID:
        suite = parse_hybrid(key)
        return (suite.post_quantum,) if suite else ()
    return ()


def default_hybrid_for(algorithm: str) -> str | None:
    """The registered hybrid that keeps this primitive as its classical half."""
    key = normalize(algorithm)
    for suite in HYBRID_SUITES.values():
        if suite.classical == key:
            return suite.name
    pure = _pure_replacement(key)
    return f"{key}{SEPARATOR}{pure}" if pure else None


def _pure_replacement(algorithm: str) -> str | None:
    from .algorithms import default_replacement

    return default_replacement(algorithm)


def ladder_replacements(algorithms: Mapping[str, str] | None = None) -> dict[str, str]:
    """A replacement map that migrates classical primitives to their hybrid.

    Supplied as an alternative to the pure post-quantum map so an experiment can
    compare the two policies rather than assuming one.
    """
    out: dict[str, str] = {}
    for name in ("RSA-2048", "RSA-3072", "RSA-4096", "ECDSA", "ECDH", "X25519",
                 "ED25519", "DH", "DSA"):
        hybrid = default_hybrid_for(name)
        if hybrid:
            out[name] = hybrid
    if algorithms:
        out.update(algorithms)
    return out
