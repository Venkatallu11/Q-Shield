"""Personal identity: how much of the risk is actually yours to fix?

Enterprise migration planning assumes the planner owns the assets. An individual
does not. You can move your own mail provider, replace a passkey, or change where
your backups live. You cannot migrate your bank's key exchange, your government's
signing certificate, or the archive holding your medical records. Any plan that
quietly treats those as budget items reports a risk reduction that no one can
buy.

This experiment therefore does not ask "which planner wins" — with a fixed
feasible set that comparison is structurally determined, as it was in 0.3. It
decomposes the modeled risk instead:

``controllable_share``
    the fraction of total modeled risk carried by assets the individual can act
    on at all.
``addressable_share``
    the fraction that a *given budget* can actually remove.
``residual_after_best_plan``
    what is left once the individual has done everything available to them.

The split by threat class matters just as much. Credential risk
(``AUTHENTICATION``) is remediable: a forged credential is a live attack, and
rotating or replacing it later still helps. Harvested-data risk
(``CONFIDENTIALITY``) is not: ciphertext copied today is copied, and migrating
afterwards protects only future traffic. Where an individual's residual risk sits
between those two classes decides whether "do it later" is a reasonable answer.

**Scope, stated plainly.** Q-SHIELD models quantum risk and nothing else. It does
not model phishing, credential stuffing, SIM-swap, malware or device theft, which
between them account for the overwhelming majority of real personal identity
compromise. Nothing here should be read as a personal security assessment. See
docs/IDENTITY.md.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from ..model import DEFAULT_WEIGHTS, AssetModel, ObjectiveWeights
from ..optimizer import MigrationProblem, exhaustive
from ..paths import EdgeKind, EdgeModel
from ..threat import ThreatClass
from ..uncertainty import summarize

REPLACEMENTS: dict[str, str] = {
    "X25519": "ML-KEM",
    "ECDH": "ML-KEM",
    "RSA-2048": "ML-KEM",
    "ED25519": "ML-DSA",
    "ECDSA": "ML-DSA",
}


@dataclass(frozen=True)
class PersonalConfig:
    """Shape of a synthetic personal identity estate.

    Defaults reflect the ordinary case: a handful of accounts the person
    genuinely controls, and rather more third-party services holding data about
    them that they do not.
    """

    controlled_accounts: int = 4
    third_party_services: int = 5
    long_lived_records: int = 2      # health, genomic, financial history
    budget_low: float = 1.0
    budget_high: float = 4.0

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def make_instance(
    rng: random.Random, config: PersonalConfig = PersonalConfig()
) -> MigrationProblem:
    assets: list[AssetModel] = []
    edges: list[EdgeModel] = []

    idp = "identity-provider"
    assets.append(
        AssetModel(
            idp, rng.choice(("ECDSA", "ED25519")), sensitivity=rng.uniform(0.9, 1.0),
            exposure=rng.uniform(0.5, 0.7), data_lifetime_years=10,
            dependency_count=config.controlled_accounts + config.third_party_services,
            rotation_hours=168, migration_cost=2.0, business_criticality=1.0,
            controllable=False,  # you choose a provider; you do not migrate one
            threat_class=ThreatClass.AUTHENTICATION, credential_validity_years=5,
        )
    )

    controlled = [f"account-{i}" for i in range(config.controlled_accounts)]
    for name in controlled:
        signing = rng.random() < 0.5
        assets.append(
            AssetModel(
                name,
                rng.choice(("ED25519", "ECDSA")) if signing else rng.choice(("X25519", "RSA-2048")),
                sensitivity=rng.uniform(0.6, 0.95), exposure=rng.uniform(0.7, 1.0),
                data_lifetime_years=rng.uniform(5, 20),
                dependency_count=rng.randint(2, 6),
                rotation_hours=rng.uniform(24, 168),
                migration_cost=rng.uniform(0.5, 1.5), business_criticality=0.8,
                controllable=True,
                threat_class=ThreatClass.AUTHENTICATION if signing else ThreatClass.CONFIDENTIALITY,
                credential_validity_years=rng.uniform(1, 5) if signing else None,
            )
        )

    third_party = [f"service-{i}" for i in range(config.third_party_services)]
    for name in third_party:
        assets.append(
            AssetModel(
                name, rng.choice(("ECDH", "X25519", "RSA-2048")),
                sensitivity=rng.uniform(0.7, 1.0), exposure=rng.uniform(0.4, 0.8),
                data_lifetime_years=rng.uniform(7, 20), dependency_count=rng.randint(2, 5),
                rotation_hours=rng.uniform(48, 720), migration_cost=1.0,
                business_criticality=0.9, controllable=False,
                threat_class=ThreatClass.CONFIDENTIALITY,
            )
        )

    records = [f"record-{i}" for i in range(config.long_lived_records)]
    for name in records:
        assets.append(
            AssetModel(
                name, "RSA-2048", sensitivity=1.0, exposure=rng.uniform(0.25, 0.45),
                # Health and genomic data outlive every enterprise horizon, and
                # genomic data implicates relatives who never consented at all.
                data_lifetime_years=rng.uniform(40, 100),
                dependency_count=rng.randint(1, 3), rotation_hours=0.0,
                migration_cost=1.0, business_criticality=1.0, controllable=False,
                threat_class=ThreatClass.CONFIDENTIALITY,
            )
        )

    # The identity provider is a trust anchor: compromising it compromises every
    # account federated through it, at once. Structurally a root CA.
    for name in controlled + third_party:
        if rng.random() < 0.7:
            edges.append(EdgeModel(idp, name, rng.uniform(0.85, 1.0), EdgeKind.DELEGATION))

    # Account-recovery chains are the attack paths: mail resets the account that
    # unlocks the service that holds the record.
    chain = [controlled[0], *controlled[1:], *third_party, *records]
    for u, v in zip(chain, chain[1:], strict=False):
        edges.append(EdgeModel(u, v, rng.uniform(0.5, 0.95), EdgeKind.DEPENDENCY))
    for _ in range(3):
        u, v = rng.choice(controlled), rng.choice(third_party + records)
        if u != v:
            edges.append(EdgeModel(u, v, rng.uniform(0.3, 0.8), EdgeKind.DEPENDENCY))

    return MigrationProblem(
        assets=tuple(assets), edges=tuple(edges),
        entrypoints=(controlled[0],), targets=(records[-1],),
        replacements=REPLACEMENTS,
        budget=rng.uniform(config.budget_low, config.budget_high),
    )


def make_instances(
    count: int, seed: int, config: PersonalConfig = PersonalConfig()
) -> list[MigrationProblem]:
    rng = random.Random(seed)
    return [make_instance(rng, config) for _ in range(count)]


def run(
    *,
    instances: int = 300,
    seed: int = 20260907,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    config: PersonalConfig = PersonalConfig(),
) -> dict[str, object]:
    problems = [p for p in make_instances(instances, seed, config) if p.candidates]

    controllable_share: list[float] = []
    addressable_share: list[float] = []
    residual: list[float] = []
    by_class: dict[str, list[float]] = {c.value: [] for c in ThreatClass}
    residual_by_class: dict[str, list[float]] = {c.value: [] for c in ThreatClass}

    for problem in problems:
        baseline = problem.evaluate((), weights=weights)
        best, _ = exhaustive(problem, weights=weights)
        after = problem.evaluate(best.selected, weights=weights)

        own = baseline.own_scores
        total = sum(own.values()) or 1.0
        controllable_share.append(
            sum(own[a.name] for a in problem.assets if a.controllable) / total
        )
        addressable_share.append(
            (baseline.value - after.value) / baseline.value if baseline.value > 0 else 0.0
        )
        residual.append(after.value)

        # Where does the risk sit, before and after doing everything available?
        migrated = {a.name: a for a in problem.assets}
        for asset in problem.assets:
            cls = asset.effective_threat_class.value
            by_class[cls].append(own[asset.name])
        after_assets = {a.name: a for a in problem.assets}
        for name, score in after.own_scores.items():
            residual_by_class[after_assets[name].effective_threat_class.value].append(score)
        del migrated

    return {
        "experiment": "personal_identity",
        "question": (
            "For an individual, how much modeled quantum risk is theirs to act "
            "on, and what is left once they have done everything they can?"
        ),
        "design": {
            "instances": len(problems),
            "not_a_win_rate": (
                "no strategy comparison: with a fixed feasible set the exhaustive "
                "planner cannot lose. This decomposes the risk instead."
            ),
            "scope": (
                "quantum risk only. Phishing, credential stuffing, SIM-swap, malware "
                "and device theft are not modelled and dominate real personal "
                "identity compromise."
            ),
            "weights": weights.as_tuple(),
            "seed": seed,
            "generator": config.as_dict(),
        },
        "share_of_risk_on_assets_the_individual_controls": summarize(controllable_share),
        "share_of_objective_removable_within_budget": summarize(addressable_share),
        "residual_objective_after_best_available_plan": summarize(residual),
        "risk_by_threat_class": {
            cls: summarize(values) for cls, values in by_class.items() if values
        },
        "residual_risk_by_threat_class": {
            cls: summarize(values) for cls, values in residual_by_class.items() if values
        },
        "interpretation": (
            "CONFIDENTIALITY residual is the part that cannot be deferred: data "
            "harvested today is harvested, and migrating later protects only future "
            "traffic. AUTHENTICATION residual is remediable later, because forgery "
            "is a live attack rather than a retroactive one."
        ),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
