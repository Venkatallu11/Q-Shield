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


# --- 0.5 experiments --------------------------------------------------------

PERSONAL_CASE = Path(__file__).resolve().parents[1] / "cases" / "personal_identity_case.json"


def test_personal_case_loads_with_threat_classes_and_control_flags():
    from qshield.threat import ThreatClass

    problem = benchmark.load_case(PERSONAL_CASE)
    names = {a.name for a in problem.assets}
    assert {"identity-provider", "genomic-data", "passkey"} <= names
    assert problem.uncontrollable, "third-party assets must be marked uncontrollable"
    assert all(a.controllable for a in problem.candidates)
    genome = next(a for a in problem.assets if a.name == "genomic-data")
    assert genome.effective_threat_class is ThreatClass.CONFIDENTIALITY


def test_case_loader_accepts_underscore_comment_keys():
    """Case files carry provenance and caveats inline; the loader must not choke."""
    problem = benchmark.problem_from_dict(
        {
            "_comment": "documentation, not data",
            "assets": [{"_note": "x", "name": "a", "algorithm": "RSA-2048"}],
            "edges": [],
            "entrypoints": ["a"],
            "targets": ["a"],
            "replacements": {},
            "budget": 1,
        }
    )
    assert problem.assets[0].name == "a"


def test_case_loader_resolves_edge_kind_from_a_string():
    from qshield.paths import EdgeKind

    problem = benchmark.problem_from_dict(
        {
            "assets": [{"name": "r", "algorithm": "ECDSA"}, {"name": "l", "algorithm": "ECDSA"}],
            "edges": [{"source": "r", "target": "l", "kind": "delegation"}],
            "entrypoints": ["r"],
            "targets": ["l"],
            "replacements": {},
            "budget": 1,
        }
    )
    assert problem.edges[0].kind is EdgeKind.DELEGATION


@pytest.mark.slow
def test_tail_risk_sweep_includes_the_0_4_anchor_and_is_reproducible():
    from qshield.experiments import tail_risk

    report = tail_risk.run(instances=10, alphas=(1.0, 0.25))
    anchors = [r for r in report["sweep"] if r["is_0_4_behaviour"]]
    assert len(anchors) == 1, "alpha=1 must be present as the 0.4 baseline"
    assert tail_risk.run(instances=10, alphas=(1.0, 0.25)) == report


@pytest.mark.slow
def test_hierarchy_reports_error_size_not_a_win_rate():
    from qshield.experiments import hierarchy

    report = hierarchy.run(instances=10)
    assert "not_a_win_rate" in report["design"]
    aware = report["per_arm"]["hierarchy_aware"]
    # The aware planner's own plan cannot be illusory: it is scored in its model.
    assert aware["illusory_fraction"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert aware["wasted_spend"]["mean"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.slow
def test_hierarchy_root_cost_sweep_bounds_the_claim():
    from qshield.experiments import hierarchy

    report = hierarchy.sweep_root_cost(instances=8, root_costs=(0.5, 3.0))
    cheap, dear = report["sweep"]
    # Cheap anchors get bought anyway, so the modelling refinement matters less.
    assert cheap["blind_illusory_fraction"] < dear["blind_illusory_fraction"]


@pytest.mark.slow
def test_threat_class_experiment_reports_tier_selection():
    from qshield.experiments import threat_class

    report = threat_class.run(instances=10)
    tiers = report["assets_selected_by_tier"]
    assert set(tiers) == {"root", "intermediate", "leaf"}
    assert 0.0 <= report["plan_agreement"] <= 1.0


def test_pre_0_5_view_discards_the_class_specific_horizons():
    from qshield.experiments.pki import make_instances as pki_instances
    from qshield.experiments.threat_class import as_pre_0_5
    from qshield.threat import ThreatClass

    problem = pki_instances(1, 5)[0]
    legacy = as_pre_0_5(problem)
    assert all(a.credential_validity_years is None for a in legacy.assets)
    assert all(
        a.effective_threat_class is ThreatClass.CONFIDENTIALITY for a in legacy.assets
    )
    # Same graph, same budget: only the scoring differs.
    assert legacy.edges == problem.edges and legacy.budget == problem.budget


@pytest.mark.slow
def test_personal_experiment_decomposes_rather_than_ranking():
    from qshield.experiments import personal

    report = personal.run(instances=10)
    assert "not_a_win_rate" in report["design"]
    assert "scope" in report["design"]
    share = report["share_of_risk_on_assets_the_individual_controls"]["mean"]
    assert 0.0 <= share <= 1.0


def test_pki_generator_builds_a_real_hierarchy():
    from qshield.experiments.pki import delegation_as_dependency
    from qshield.experiments.pki import make_instances as pki
    from qshield.paths import EdgeKind

    problem = pki(1, 11)[0]
    assert any(e.kind is EdgeKind.DELEGATION for e in problem.edges)
    result = problem.evaluate(())
    leaves = [a.name for a in problem.assets if a.name.startswith("leaf")]
    # Every leaf inherits from its issuer, so effective exceeds own risk.
    assert any(result.node_scores[n] > result.own_scores[n] + 1e-9 for n in leaves)
    # The blind view has no delegation edges left.
    assert all(e.kind is EdgeKind.DEPENDENCY for e in delegation_as_dependency(problem).edges)


def test_pki_replacements_map_signing_keys_to_signature_algorithms():
    """RSA defaults to ML-KEM because key transport is the usual driver; a
    signing hierarchy must override that."""
    from qshield.experiments.pki import REPLACEMENTS

    assert REPLACEMENTS["RSA-2048"] == "ML-DSA"
    assert set(REPLACEMENTS.values()) == {"ML-DSA"}
