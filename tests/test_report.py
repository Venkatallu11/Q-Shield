"""The Markdown report.

Its job is to be read by a person who will spend money on the strength of it, so
the tests are mostly about whether the caveats survive rendering.
"""

from pathlib import Path

import pytest

from qshield.experiments.benchmark import load_case
from qshield.experiments.robustness import run as run_robustness
from qshield.model import AssetModel
from qshield.optimizer import MigrationProblem
from qshield.paths import EdgeKind, EdgeModel
from qshield.planner import choose
from qshield.report import render, render_plan_only
from qshield.threat import ThreatClass

CASES = Path(__file__).resolve().parents[1] / "cases"


@pytest.fixture
def pki_problem():
    return load_case(CASES / "pki_case.json")


def test_renders_markdown_with_the_expected_sections(pki_problem):
    text = render(pki_problem)
    for heading in ("# ", "## Recommendation", "## Before you believe any of this"):
        assert heading in text


def test_the_recommendation_comes_before_the_caveats(pki_problem):
    """A caveat nobody scrolls to is not a caveat, but burying the answer makes
    the report unusable. Recommendation first, caveats immediately after."""
    text = render(pki_problem)
    assert text.index("## Recommendation") < text.index("## Before you believe")
    assert text.index("## Before you believe") < text.index("## Inventory")


def test_scope_limits_are_always_stated(pki_problem):
    text = render(pki_problem)
    assert "uncalibrated" in text
    assert "phishing" in text  # quantum risk only


def test_provenance_caveats_reach_the_reader(pki_problem):
    provenance = {
        "summary": {"assumed_share": 0.55, "always_assumed": ["sensitivity", "exposure"]},
        "warnings": ["3 certificates have already expired"],
    }
    text = render(pki_problem, provenance=provenance)
    assert "55% of the values behind these numbers were not observed" in text
    assert "`sensitivity`" in text
    assert "3 certificates have already expired" in text


def test_an_approximate_plan_says_so():
    assets = tuple(
        AssetModel(f"a{i}", "RSA-2048", 0.8, 0.8, 10, migration_cost=1.0)
        for i in range(20)
    )
    problem = MigrationProblem(assets, (), (), (), {"RSA-2048": "ML-KEM"}, 5.0)
    text = render(problem)
    assert "approximate" in text
    assert "not tractable" in text


def test_inherited_risk_gets_its_own_section(pki_problem):
    """The crypto-agility trap, stated for the estate in front of the reader."""
    text = render(pki_problem)
    assert "## Assets that cannot be fixed on their own" in text
    assert "the spend is real and the risk reduction is zero" in text
    assert "Migrate the 2 issuing authorities first." in text


def test_no_inheritance_section_when_there_is_no_hierarchy():
    assets = (AssetModel("solo", "RSA-2048", 0.8, 0.8, 10, migration_cost=1.0),)
    problem = MigrationProblem(assets, (), (), (), {"RSA-2048": "ML-KEM"}, 2.0)
    assert "cannot be fixed on their own" not in render(problem)


def test_an_inert_path_term_is_explained():
    """Reporting 'attack paths: 0' without saying what follows invites the reader
    to assume the path analysis ran and found nothing."""
    assets = (
        AssetModel("root", "ECDSA", 0.8, 0.3, 10, migration_cost=1.0,
                   threat_class=ThreatClass.AUTHENTICATION,
                   credential_validity_years=20),
        AssetModel("leaf", "ECDSA", 0.8, 0.9, 10, migration_cost=0.5,
                   threat_class=ThreatClass.AUTHENTICATION,
                   credential_validity_years=0.25),
    )
    edges = (EdgeModel("root", "leaf", 1.0, EdgeKind.DELEGATION),)
    problem = MigrationProblem(assets, edges, (), (), {"ECDSA": "ML-DSA"}, 2.0)
    assert "the path term is inert" in render(problem)


def test_robustness_verdicts_are_rendered_as_a_work_queue(pki_problem):
    report = run_robustness(pki_problem, draws=40, seed=2)
    text = render(pki_problem, robustness=report)
    assert "| asset | chosen in | verdict |" in text
    assert "redraws of every guessed parameter" in text


def test_without_robustness_the_report_says_to_run_it(pki_problem):
    assert "Run `qshield robustness`" in render(pki_problem)


def test_budget_frontier_marks_the_current_budget(pki_problem):
    text = render(pki_problem)
    assert "## What more budget would buy" in text
    assert "current budget" in text


def test_inventory_reports_the_horizon_that_was_actually_used(pki_problem):
    """An AUTHENTICATION asset is scored on credential validity, not retention;
    showing the retention figure next to its score would misexplain it."""
    text = render(pki_problem)
    assert "in service" in text
    assert "verifiable" in text  # the code-signing certificate


def test_plan_only_summary_is_a_single_line(pki_problem):
    line = render_plan_only(choose(pki_problem))
    assert "\n" not in line and "migrate" in line


def test_report_ends_with_provenance_of_the_tool_itself(pki_problem):
    from qshield import __version__

    assert __version__ in render(pki_problem)
