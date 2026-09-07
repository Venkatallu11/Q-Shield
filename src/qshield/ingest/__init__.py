"""Turn real artifacts into Q-SHIELD cases.

Every experiment before 0.6 ran on synthetic instances, which meant the model
could not be pointed at anything real. These adapters close that gap, and carry a
provenance record so that the values an artifact genuinely supplies are never
confused with the ones filled in to make the model run.
"""

from .provenance import AssetProvenance, InventoryProvenance, Source

__all__ = ["AssetProvenance", "InventoryProvenance", "Source"]
