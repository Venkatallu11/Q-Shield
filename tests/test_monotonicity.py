"""Monotonicity is the property the optimizer's correctness rests on.

If migrating an asset to a strictly less susceptible algorithm can *increase* the
objective, then the planner is penalised for beneficial actions and every result
built on the objective is suspect. 0.3 violated this. These tests pin it down by
construction and over randomised instances.
"""

import random

import pytest

from qshield.algorithms import default_replacement, quantum_factor
from qshield.experiments.generator import REPLACEMENTS, make_instances
from qshield.model import AssetModel, ObjectiveWeights, apply_migration, objective
from qshield.paths import EdgeModel


def test_documented_0_3_counterexample_is_monotone():
    """The exact inventory from the 0.3 regression: migrating 'a' to ML-KEM
    raised the objective from 18.838393 to 19.329375 because 'a' fell below the
    rotation-pressure gate and stopped diluting the mean."""
    a = AssetModel("a", "RSA-2048", 0.50, 0.44, 20, 1, 4.0)
    b = AssetModel("b", "RSA-2048", 0.95, 0.95, 20, 1, 168.0)
    before = objective([a, b], []).value
    after = objective(apply_migration([a, b], ["a"], {"RSA-2048": "ML-KEM"}), []).value
    assert after < before


@pytest.mark.parametrize("seed", range(12))
def test_any_single_migration_is_never_harmful(seed):
    """Over random instances, no single migration may raise the objective."""
    problem = make_instances(1, seed)[0]
    base = problem.evaluate(()).value
    for asset in problem.candidates:
        assert problem.evaluate([asset.name]).value <= base + 1e-9, asset.name


@pytest.mark.parametrize("seed", range(8))
def test_every_feasible_plan_is_never_harmful(seed):
    """Stronger: no *combination* of migrations may raise the objective either."""
    problem = make_instances(1, seed + 100)[0]
    base = problem.evaluate(()).value
    for names, _cost in problem.feasible_plans():
        assert problem.evaluate(names).value <= base + 1e-9, names


@pytest.mark.parametrize("seed", range(6))
def test_monotone_under_arbitrary_weightings(seed):
    """Monotonicity must not depend on the weights, since they are uncalibrated."""
    rng = random.Random(seed)
    problem = make_instances(1, seed + 200)[0]
    for _ in range(5):
        weights = ObjectiveWeights(
            rng.uniform(0.01, 1), rng.uniform(0.0, 1), rng.uniform(0.0, 1)
        )
        base = problem.evaluate((), weights=weights).value
        for asset in problem.candidates:
            assert problem.evaluate([asset.name], weights=weights).value <= base + 1e-9


def test_adding_a_migration_never_increases_the_objective():
    """Superset plans dominate subset plans: a consequence of monotonicity that
    the exhaustive planner relies on when it enumerates the empty plan."""
    problem = make_instances(1, 7)[0]
    names = [a.name for a in problem.candidates]
    running: list[str] = []
    previous = problem.evaluate(()).value
    for name in names:
        running.append(name)
        current = problem.evaluate(running).value
        assert current <= previous + 1e-9
        previous = current


def test_objective_is_monotone_in_the_quantum_factor_directly():
    a = AssetModel("a", "RSA-2048", 0.8, 0.8, 15, 3, 24.0)
    b = AssetModel("b", "RSA-2048", 0.6, 0.9, 12, 2, 72.0)
    edges = [EdgeModel("a", "b", 0.9)]
    previous = None
    for qf in (1.0, 0.7, 0.4, 0.15, 0.05):
        value = objective([a, b], edges, ["a"], ["b"], qf_overrides={"a": qf, "b": qf}).value
        if previous is not None:
            assert value <= previous + 1e-12
        previous = value


def test_generator_replacements_are_all_strict_improvements():
    """Monotonicity only implies migration helps if replacements really are less
    susceptible; this guards the generator's replacement map against drift."""
    for source, target in REPLACEMENTS.items():
        assert quantum_factor(target) < quantum_factor(source)
        assert default_replacement(source) == target


# --- 0.5 additions: the property must survive trust inheritance and CVaR -----


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.25, 0.05])
def test_monotone_under_every_tail_parameter(alpha):
    """CVaR is an average of the k largest values, so it is non-decreasing in
    each; the objective's monotonicity must not depend on alpha."""
    from qshield.model import ObjectiveWeights

    problem = make_instances(1, 900)[0]
    weights = ObjectiveWeights(0.55, 0.30, 0.15, alpha)
    base = problem.evaluate((), weights=weights).value
    for names, _cost in problem.feasible_plans():
        assert problem.evaluate(names, weights=weights).value <= base + 1e-9


@pytest.mark.parametrize("seed", range(8))
def test_monotone_across_a_trust_hierarchy(seed):
    """Effective risk is a max over products of monotone terms, so inheritance
    preserves the property -- including where a migration changes nothing."""
    from qshield.experiments.pki import make_instances as pki_instances

    problem = pki_instances(1, seed + 1000)[0]
    base = problem.evaluate(()).value
    for names, _cost in problem.feasible_plans():
        assert problem.evaluate(names).value <= base + 1e-9, names


def test_migrating_an_issuer_helps_at_least_as_much_as_migrating_its_leaves():
    """The operational form of the crypto-agility trap: under trust inheritance
    an anchor dominates everything beneath it."""
    from qshield.experiments.pki import make_instances as pki_instances

    for seed in range(6):
        problem = pki_instances(1, seed + 2000)[0]
        base = problem.evaluate(()).value
        roots = [a.name for a in problem.assets if a.name.startswith("root")]
        leaves = [a.name for a in problem.assets if a.name.startswith("leaf")]
        root_gain = base - problem.evaluate(roots).value
        leaf_gain = base - problem.evaluate(leaves).value
        assert root_gain >= leaf_gain - 1e-9


# --- 0.7: the calibrated path deliberately gives the guarantee up -----------


def test_uncalibrated_monotonicity_is_unaffected_by_the_calibration_work():
    """The guarantee documented in qshield.model still holds on the default path;
    calibration is opt-in and changes nothing unless asked for."""
    problem = make_instances(1, 1200)[0]
    base = problem.evaluate(()).value
    for names, _cost in problem.feasible_plans():
        assert problem.evaluate(names).value <= base + 1e-9


def test_calibrated_migration_of_a_short_lived_credential_raises_the_score():
    """Not a defect. Over a short horizon the probability that a CRQC arrives at
    all is smaller than the residual risk of a young post-quantum primitive, so
    the model says leave a 90-day certificate alone. The uncalibrated model
    multiplied a quantum factor of 1.00 by a longevity term and so could never
    express this. Documented in docs/CALIBRATION.md."""
    from qshield.calibration import DEFAULT_CALIBRATION
    from qshield.model import AssetModel, apply_migration, objective
    from qshield.threat import ThreatClass

    leaf = AssetModel(
        "leaf", "ECDSA", 0.8, 0.9, 15, 2, 24.0,
        threat_class=ThreatClass.AUTHENTICATION, credential_validity_years=0.25,
    )
    before = objective([leaf], [], calibration=DEFAULT_CALIBRATION).value
    after = objective(
        apply_migration([leaf], ["leaf"], {"ECDSA": "ML-DSA"}),
        [],
        calibration=DEFAULT_CALIBRATION,
    ).value
    assert after > before


def test_calibrated_migration_of_a_long_lived_anchor_still_helps():
    from qshield.calibration import DEFAULT_CALIBRATION
    from qshield.model import AssetModel, apply_migration, objective
    from qshield.threat import ThreatClass

    root = AssetModel(
        "root", "ECDSA", 0.9, 0.3, 15, 8, 0.0,
        threat_class=ThreatClass.AUTHENTICATION, credential_validity_years=20,
    )
    before = objective([root], [], calibration=DEFAULT_CALIBRATION).value
    after = objective(
        apply_migration([root], ["root"], {"ECDSA": "ML-DSA"}),
        [],
        calibration=DEFAULT_CALIBRATION,
    ).value
    assert after < before


@pytest.mark.parametrize("seed", range(6))
def test_planners_still_behave_without_the_monotonicity_guarantee(seed):
    """Exhaustive enumeration includes the empty plan and greedy stops when no
    move helps, so neither relies on monotonicity -- only the theorem does."""
    from dataclasses import replace as _replace

    from qshield.calibration import DEFAULT_CALIBRATION
    from qshield.experiments.pki import make_instances as pki_instances
    from qshield.planner import plan as choose_plan

    problem = _replace(
        pki_instances(1, seed + 1300)[0], calibration=DEFAULT_CALIBRATION
    )
    chosen = choose_plan(problem)
    # Whatever it picks must be no worse than doing nothing: with the null plan
    # in the candidate set, a planner that returned a harmful plan would be wrong.
    assert chosen.objective <= problem.evaluate(()).value + 1e-9
    assert chosen.cost <= problem.budget + 1e-9
