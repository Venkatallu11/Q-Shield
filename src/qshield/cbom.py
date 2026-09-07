"""CycloneDX cryptographic bill of materials export.

Kept from 0.3 but retargeted at :class:`~qshield.model.AssetModel`. 0.3's version
imported ``engine.Asset``, a type from the abandoned 0.1 lineage that no other
0.3 module produced, so the exporter could not be called on anything the rest of
the pipeline built.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .algorithms import family, normalize, primitive, quantum_factor
from .model import AssetModel

CYCLONEDX_SPEC_VERSION = "1.6"


def to_cbom(assets: Sequence[AssetModel]) -> dict[str, Any]:
    components = []
    for a in assets:
        components.append(
            {
                "type": "cryptographic-asset",
                "name": a.name,
                "cryptoProperties": {
                    "assetType": "algorithm",
                    "algorithmProperties": {
                        "algorithmFamily": normalize(a.algorithm),
                        "primitive": primitive(a.algorithm).value,
                        "classification": family(a.algorithm).value,
                    },
                },
                # Q-SHIELD scenario parameters are namespaced as properties rather
                # than smuggled into standard CycloneDX fields, so a consumer can
                # ignore them without misreading them as spec-defined values.
                "properties": [
                    {"name": "qshield:dataLifetimeYears", "value": str(a.data_lifetime_years)},
                    {"name": "qshield:sensitivity", "value": str(a.sensitivity)},
                    {"name": "qshield:exposure", "value": str(a.exposure)},
                    {"name": "qshield:dependencyCount", "value": str(a.dependency_count)},
                    {"name": "qshield:quantumFactor", "value": str(quantum_factor(a.algorithm))},
                    {"name": "qshield:threatClass", "value": a.effective_threat_class.value},
                    {"name": "qshield:controllable", "value": str(a.controllable).lower()},
                ],
            }
        )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": CYCLONEDX_SPEC_VERSION,
        "version": 1,
        "components": components,
    }


def save_cbom(assets: Sequence[AssetModel], path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_cbom(assets), indent=2), encoding="utf-8")
