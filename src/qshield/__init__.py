"""Q-SHIELD — a falsifiable model for post-quantum migration planning.

Q-SHIELD asks whether attack-path-aware, cost-constrained migration planning
reduces modeled quantum exposure more efficiently than flat asset-priority
ranking. It is a research instrument: the objective is a scenario model with
uncalibrated parameters, not a compliance score or a security assessment.

Public surface::

    AssetModel, EdgeModel, ObjectiveWeights, objective   -- the model
    MigrationProblem, exhaustive, greedy_flat, ...       -- planners
    sample_worlds, paired_comparison                     -- uncertainty
    qshield.experiments.{benchmark,ablation,generalization,sensitivity}
"""

__version__ = "0.4.0"

from .algorithms import Family, normalize, quantum_factor
from .model import (
    DEFAULT_WEIGHTS,
    NODE_ONLY_WEIGHTS,
    AssetModel,
    ObjectiveResult,
    ObjectiveWeights,
    apply_migration,
    node_risk,
    objective,
    path_risk,
)
from .optimizer import (
    MigrationProblem,
    Plan,
    do_nothing,
    exhaustive,
    greedy_flat,
    greedy_marginal,
    pareto_frontier,
)
from .paths import EdgeModel, PathSet, enumerate_paths
from .uncertainty import (
    PerturbationModel,
    World,
    bootstrap_ci,
    paired_comparison,
    sample_worlds,
    summarize,
)

__all__ = [
    "__version__",
    "AssetModel",
    "DEFAULT_WEIGHTS",
    "EdgeModel",
    "Family",
    "MigrationProblem",
    "NODE_ONLY_WEIGHTS",
    "ObjectiveResult",
    "ObjectiveWeights",
    "PathSet",
    "PerturbationModel",
    "Plan",
    "World",
    "apply_migration",
    "bootstrap_ci",
    "do_nothing",
    "enumerate_paths",
    "exhaustive",
    "greedy_flat",
    "greedy_marginal",
    "node_risk",
    "normalize",
    "objective",
    "paired_comparison",
    "pareto_frontier",
    "path_risk",
    "quantum_factor",
    "sample_worlds",
    "summarize",
]
