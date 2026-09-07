"""Where each modelled value came from.

Q-SHIELD's objective takes nine parameters per asset. A certificate supplies
roughly half of them exactly and says nothing whatever about the rest: no
certificate records how sensitive the data behind it is, how exposed the host is,
or what migrating it would cost.

A tool that silently fills those in with defaults and prints a risk score is
worse than no tool, because the output is indistinguishable from a measurement.
Every generated asset therefore carries a provenance record, every report prints
the assumed-field count, and :mod:`qshield.experiments.robustness` exists to
answer the question this raises — whether the recommendation actually depends on
the assumptions.

Three levels, and the boundary between the last two is the one that matters:

``OBSERVED``
    Read directly out of the artifact. A validity period, a public key
    algorithm, an issuer name.

``DERIVED``
    Computed from observed facts by a rule that is not a judgement call.
    Validity years from two dates; how many certificates an authority issued;
    ``NON_REPUDIATION`` from a codeSigning extended key usage, which is what that
    EKU *means*.

``ASSUMED``
    Not determinable from the artifact at all. A placeholder chosen so the model
    can run, which the operator must review before believing any number that
    depends on it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

# Parameters no certificate, CBOM or scan can supply. Listed here rather than
# inferred from usage so that adding a model parameter forces a decision about
# where its value comes from.
NEVER_OBSERVABLE: frozenset[str] = frozenset(
    {
        "sensitivity",
        "exposure",
        "data_lifetime_years",
        "business_criticality",
        "migration_cost",
        "rotation_hours",
    }
)


class Source(str, Enum):
    OBSERVED = "observed"
    DERIVED = "derived"
    ASSUMED = "assumed"


@dataclass(frozen=True)
class FieldOrigin:
    source: Source
    # Why this value: the artifact field it was read from, the rule applied, or
    # the reason a placeholder was needed. Always populated -- a provenance
    # record with no explanation is not much better than none.
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"source": self.source.value, "detail": self.detail}


@dataclass
class AssetProvenance:
    """Per-field origins for one generated asset."""

    asset: str
    origin: str  # the artifact this asset came from: a file path, a host:port
    fields: dict[str, FieldOrigin] = field(default_factory=dict)

    def record(self, name: str, source: Source, detail: str) -> None:
        self.fields[name] = FieldOrigin(source, detail)

    def observed(self, name: str, detail: str) -> None:
        self.record(name, Source.OBSERVED, detail)

    def derived(self, name: str, detail: str) -> None:
        self.record(name, Source.DERIVED, detail)

    def assumed(self, name: str, detail: str) -> None:
        self.record(name, Source.ASSUMED, detail)

    def count(self, source: Source) -> int:
        return sum(1 for o in self.fields.values() if o.source is source)

    @property
    def assumed_fields(self) -> list[str]:
        return sorted(n for n, o in self.fields.items() if o.source is Source.ASSUMED)

    def as_dict(self) -> dict[str, object]:
        return {
            "asset": self.asset,
            "origin": self.origin,
            "fields": {n: o.as_dict() for n, o in sorted(self.fields.items())},
        }


@dataclass
class InventoryProvenance:
    """Provenance for a whole generated case, plus the caveats it implies."""

    assets: list[AssetProvenance] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, record: AssetProvenance) -> None:
        self.assets.append(record)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def summary(self) -> Mapping[str, object]:
        totals = {s.value: sum(a.count(s) for a in self.assets) for s in Source}
        graded = sum(totals.values())
        assumed_everywhere = sorted(
            {name for a in self.assets for name in a.assumed_fields}
        )
        return {
            "assets": len(self.assets),
            "fields_by_source": totals,
            "assumed_share": round(totals["assumed"] / graded, 4) if graded else 0.0,
            "always_assumed": assumed_everywhere,
            "reading": (
                "OBSERVED and DERIVED values come from the artifacts. ASSUMED values "
                "do not exist in any artifact and were filled in so the model could "
                "run. Review them, or run `qshield robustness` to find out whether "
                "the recommendation depends on them at all."
            ),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "warnings": list(self.warnings),
            "assets": [a.as_dict() for a in self.assets],
        }
