"""Experiment drivers: reproducibility, report shape, and the design invariants
the conclusions depend on."""

import json
from pathlib import Path

import pytest

from qshield.cbom import to_cbom
from qshield.cli import main
from qshield.experiments import ablation, benchmark, generalization, sensitivity
from qshield.experiments.generator import GeneratorConfig, make_instance, make_instances
from qshield.model import AssetModel

CASE = Path(__file__).resolve().parents[1] / "cases" / "reference_case.json"

FAST = dict(instances=8, trials=10)


def test_reference_case_loads_and_is_well_formed():
    problem = benchmark.load_case(CASE)
    assert len(problem.assets) == 5
    assert problem.path_set.paths
    assert {a.name for a in problem.candidates} <= {a.name for a in problem.assets}


def test_case_loader_reports_missing_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        benchmark.problem_from_dict({"assets": []})


def test_case_loader_rejects_unknown_entrypoints():
    with pytest.raises(ValueError, match="entrypoints reference unknown"):
        benchmark.problem_from_dict(
            {
                "assets": [{"name": "a", "algorithm": "RSA-2048"}],
                "edges": [],
                "entrypoints": ["ghost"],
                "targets": ["a"],
                "replacements": {},
                "budget": 1,
            }
        )


def test_benchmark_report_is_reproducible_and_self_describing():
    problem = benchmark.load_case(CASE)
    first = benchmark.run(problem, trials=25, seed=1)
    second = benchmark.run(problem, trials=25, seed=1)
    assert first == second
    # A report must carry the assumptions behind its numbers.
    assert first["assumptions"]["algorithm_registry"]
    assert first["assumptions"]["common_random_numbers"] is True
    assert first["caveats"]


def test_benchmark_plans_are_all_budget_feasible():
    problem = benchmark.load_case(CASE)
    report = benchmark.run(problem, trials=10)
    for plan in report["plans"].values():
        assert plan["cost"] <= problem.budget + 1e-9


def test_benchmark_pareto_frontier_is_increasing_in_cost():
    report = benchmark.run(benchmark.load_case(CASE), trials=10)
    costs = [p["cost"] for p in report["pareto_frontier"]]
    assert costs == sorted(costs)


@pytest.mark.slow
def test_ablation_separates_the_two_effects():
    report = ablation.run(**FAST)
    assert set(report["design"]["arms"]) == {"flat_priority", "node_only", "path_aware"}
    # The contrast that isolates path-awareness must be present and held out.
    contrast = report["contrasts"]["node_only_vs_path_aware"]
    assert contrast["path_max"]["held_out"] is True
    assert contrast["objective"]["held_out"] is False


@pytest.mark.slow
def test_ablation_never_observes_an_impossible_in_objective_loss():
    report = ablation.run(**FAST)
    assert report["structural_check"]["in_objective_losses"] == 0


@pytest.mark.slow
def test_ablation_is_reproducible():
    assert ablation.run(**FAST, seed=5) == ablation.run(**FAST, seed=5)


@pytest.mark.slow
def test_generalization_evaluates_beyond_the_training_point():
    report = generalization.run(instances=6)
    grid = report["per_weighting"]
    assert sum(1 for e in grid if e["is_training_point"]) <= 1
    assert len(grid) == report["design"]["eval_weightings"] == 26


def test_sensitivity_reports_plan_stability():
    report = sensitivity.run(benchmark.load_case(CASE))
    assert report["distinct_plans"] >= 1
    assert report["modal_plan"]["share"] <= 1.0
    # Committing to the modal plan can never beat the per-weighting optimum.
    assert report["modal_plan"]["regret_vs_per_weighting_optimum"]["mean"] >= 0
    assert report["reference_plan"]["max_regret"] >= 0


def test_generator_instances_are_reachable_and_have_candidates():
    for problem in make_instances(30, 77):
        assert problem.path_set.paths, "backbone must guarantee a path"
        assert problem.candidates


def test_generator_cross_links_never_enter_the_entry_or_leave_the_target():
    """0.3 sampled endpoints freely, so some cross-links could not lie on any
    entry-to-target path and silently reduced topological variety."""
    for problem in make_instances(40, 79):
        for edge in problem.edges:
            assert edge.target != "entry"
            assert edge.source != "target"


def test_generator_is_deterministic_given_a_seed():
    a = make_instance(__import__("random").Random(5), GeneratorConfig())
    b = make_instance(__import__("random").Random(5), GeneratorConfig())
    assert a.assets == b.assets and a.edges == b.edges


def test_cbom_export_is_valid_cyclonedx_shape():
    assets = [AssetModel("api", "RSA-2048", 0.9, 0.9, 15, 4, 24.0)]
    bom = to_cbom(assets)
    assert bom["bomFormat"] == "CycloneDX"
    component = bom["components"][0]
    assert component["cryptoProperties"]["algorithmProperties"]["algorithmFamily"] == "RSA-2048"
    # Q-SHIELD parameters are namespaced, not smuggled into spec fields.
    assert all(p["name"].startswith("qshield:") for p in component["properties"])


def test_cli_writes_a_report(tmp_path):
    out = tmp_path / "report.json"
    assert main(["benchmark", "--input", str(CASE), "--trials", "10", "--quiet",
                 "--output", str(out)]) == 0
    assert json.loads(out.read_text())["experiment"] == "benchmark"


def test_cli_accepts_custom_weights(tmp_path):
    out = tmp_path / "r.json"
    main(["benchmark", "--input", str(CASE), "--trials", "5", "--weights", "1,0,0",
          "--quiet", "--output", str(out)])
    assert json.loads(out.read_text())["assumptions"]["weights"] == [1.0, 0.0, 0.0]


def test_cli_rejects_a_malformed_weight_triple(tmp_path):
    with pytest.raises(SystemExit):
        main(["benchmark", "--input", str(CASE), "--weights", "1,0"])
