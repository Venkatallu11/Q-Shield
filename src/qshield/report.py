"""A migration report a person can act on.

Everything else in Q-SHIELD emits JSON, which is the right format for an
experiment and the wrong one for a decision. This renders the same analysis as
prose and tables: what was found, what to do first, what the answer depends on,
and — the part most tools omit — which of the numbers behind it were measured and
which were made up.

The ordering is deliberate. The recommendation comes first because that is what
the reader wants; the caveats come immediately after it rather than at the end,
because a caveat nobody scrolls to is not a caveat.
"""

from __future__ import annotations

from collections.abc import Mapping

from . import __version__
from .model import DEFAULT_WEIGHTS, ObjectiveWeights
from .optimizer import MigrationProblem, exhaustive, pareto_frontier
from .planner import PlanningReport, choose
from .threat import ThreatClass


def render(
    problem: MigrationProblem,
    *,
    title: str = "Post-quantum migration report",
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    robustness: Mapping[str, object] | None = None,
    provenance: Mapping[str, object] | None = None,
) -> str:
    """Render the report as Markdown."""
    decision = choose(problem, weights=weights)
    baseline = problem.evaluate((), weights=weights)
    after = problem.evaluate(decision.plan.selected, weights=weights)

    lines: list[str] = [f"# {title}", ""]
    lines += _summary(problem, decision, baseline, after)
    lines += _caveats(problem, provenance, decision)
    lines += _actions(problem, decision, robustness, weights)
    lines += _inheritance(problem, baseline)
    lines += _budget(problem, weights, decision)
    lines += _inventory(problem, baseline)
    lines += _footer()
    return "\n".join(lines).rstrip() + "\n"


def _summary(problem, decision, baseline, after) -> list[str]:
    reduction = baseline.value - after.value
    share = reduction / baseline.value if baseline.value > 0 else 0.0
    plan = ", ".join(f"`{n}`" for n in decision.plan.selected) or "_nothing_"
    return [
        "## Recommendation",
        "",
        f"Migrate {plan}.",
        "",
        f"- **Spend**: {decision.plan.cost:.2f} of a {problem.budget:.2f} budget",
        f"- **Modeled risk**: {baseline.value:.2f} to {after.value:.2f} "
        f"(a {share:.0%} reduction)",
        f"- **Assets**: {len(problem.assets)} in scope, "
        f"{len(problem.candidates)} migratable, "
        f"{len(problem.uncontrollable)} outside your control",
        f"- **Attack paths**: {len(problem.path_set)}"
        + (
            " — none, so the path term is inert and only node risk and trust "
            "inheritance are scored"
            if not problem.path_set
            else ""
        ),
        f"- **Planner**: {decision.planner}"
        + ("" if decision.exact else " (approximate)"),
        "",
    ]


def _caveats(problem, provenance, decision) -> list[str]:
    lines = ["## Before you believe any of this", ""]
    if provenance:
        summary = provenance.get("summary", {})  # type: ignore[union-attr]
        share = summary.get("assumed_share", 0.0)
        assumed = summary.get("always_assumed", [])
        lines += [
            f"**{share:.0%} of the values behind these numbers were not observed.** "
            "They were filled with placeholders so the model could run:",
            "",
            *[f"- `{name}`" for name in assumed],
            "",
            "Nothing in a certificate records them. Run `qshield robustness` — "
            "already summarised below if present — to find out whether the "
            "recommendation actually depends on them.",
            "",
        ]
        for warning in provenance.get("warnings", []):  # type: ignore[union-attr]
            lines.append(f"> {warning}")
            lines.append("")
    if not decision.exact:
        lines += [
            f"> {decision.reason}",
            "",
        ]
    lines += [
        "This is a scenario model. Its quantum factors and objective weights are "
        "uncalibrated hypotheses, not measurements, and a lower score does not "
        "mean a system is safe. It models quantum risk only — not phishing, "
        "credential theft, misconfiguration or key mismanagement, which are the "
        "likelier ways any of these assets is actually compromised.",
        "",
    ]
    return lines


def _actions(problem, decision, robustness, weights) -> list[str]:
    lines = ["## What to do first", ""]
    if robustness:
        act = robustness.get("act_now", [])
        blocked = robustness.get("blocked_by_budget", [])
        contested = robustness.get("contested", [])
        stable = robustness.get("nominal_plan_reproduced_in_draws", 0.0)
        lines += [
            f"Across {robustness['design']['draws']} redraws of every guessed "  # type: ignore[index]
            f"parameter, this exact plan came back {stable:.0%} of the time.",
            "",
            "| asset | chosen in | verdict |",
            "|---|---|---|",
        ]
        for entry in robustness["assets"]:  # type: ignore[index]
            if entry["selection_rate"] < 0.01:
                continue
            lines.append(
                f"| `{entry['asset']}` | {entry['selection_rate']:.0%} | "
                f"{entry['verdict']} |"
            )
        lines.append("")
        if act:
            lines += [
                "**Act now.** These are selected under nearly every admissible "
                "setting of the guessed parameters, so calibrating them first "
                "would not change the answer: "
                + ", ".join(f"`{n}`" for n in act)
                + ".",
                "",
            ]
        unaffordable = robustness.get("never_affordable", [])
        if unaffordable:
            lines += [
                "**Never affordable.** No draw could fit these in the budget, so "
                "they were not assessed — this is not a finding that they do not "
                "matter: "
                + ", ".join(f"`{n}`" for n in unaffordable)
                + ".",
                "",
            ]
        if blocked:
            lines += [
                "**Blocked by budget.** These would be chosen whenever they are "
                "affordable. That is a funding finding, not a cryptographic one: "
                + ", ".join(f"`{n}`" for n in blocked)
                + ".",
                "",
            ]
        if contested:
            lines += [
                "**Genuinely open.** These flip with the guessed parameters. "
                "Measure their sensitivity, exposure or migration cost rather "
                "than letting a placeholder decide: "
                + ", ".join(f"`{n}`" for n in contested)
                + ".",
                "",
            ]
        likely = [
            e["asset"]
            for e in robustness["assets"]  # type: ignore[index]
            if e["verdict"] == "likely"
        ]
        if likely:
            lines += [
                "**Probably right, not certain.** Chosen in most but not all "
                "admissible settings — reasonable to start on, worth confirming: "
                + ", ".join(f"`{n}`" for n in likely)
                + ".",
                "",
            ]
        if not act and not blocked and not likely:
            lines += [
                "**No recommendation survives the uncertainty.** Every candidate "
                "depends on parameters that were guessed. Measure them before "
                "committing budget.",
                "",
            ]
    else:
        lines += [
            "Run `qshield robustness` on this case to find out which of these "
            "recommendations survive the parameters that had to be guessed.",
            "",
        ]
    return lines


def _inheritance(problem, baseline) -> list[str]:
    """The crypto-agility trap, stated for this specific estate."""
    inherited = {
        name: baseline.node_scores[name] - baseline.own_scores[name]
        for name in baseline.node_scores
        if baseline.node_scores[name] - baseline.own_scores[name] > 1e-9
    }
    if not inherited:
        return []
    issuers = {e.source for e in problem.edges if e.kind.value == "delegation"}
    lines = [
        "## Assets that cannot be fixed on their own",
        "",
        f"{len(inherited)} asset(s) inherit more risk from the authority that "
        "issued them than they carry themselves. Migrating them while their "
        "issuer still uses a classical algorithm changes nothing at all — the "
        "spend is real and the risk reduction is zero.",
        "",
        "| asset | own risk | inherited | issuer |",
        "|---|---|---|---|",
    ]
    parents = {
        e.target: e.source for e in problem.edges if e.kind.value == "delegation"
    }
    for name, gap in sorted(inherited.items(), key=lambda kv: -kv[1]):
        lines.append(
            f"| `{name}` | {baseline.own_scores[name]:.1f} | +{gap:.1f} | "
            f"`{parents.get(name, '?')}` |"
        )
    lines += [
        "",
        f"Migrate the {len(issuers)} issuing authorit"
        f"{'y' if len(issuers) == 1 else 'ies'} first.",
        "",
    ]
    return lines


def _budget(problem, weights, decision) -> list[str]:
    """What each additional unit of budget would buy."""
    if len(problem.candidates) > 16:
        # The frontier needs the full enumeration, which is not tractable here.
        return []
    _, all_plans = exhaustive(problem, weights=weights)
    frontier = pareto_frontier(all_plans)
    if len(frontier) < 2:
        return []
    lines = [
        "## What more budget would buy",
        "",
        "| spend | modeled risk | migrate |",
        "|---|---|---|",
    ]
    for point in frontier:
        chosen = ", ".join(f"`{n}`" for n in point.selected) or "_nothing_"
        marker = " **<- current budget**" if point.cost <= problem.budget else ""
        lines.append(f"| {point.cost:.2f} | {point.objective:.2f} | {chosen}{marker} |")
    lines += [
        "",
        "Rows above your budget are what a larger one would reach. This frontier "
        "uses the same guessed parameters as everything else.",
        "",
    ]
    return lines


def _inventory(problem, baseline) -> list[str]:
    lines = [
        "## Inventory",
        "",
        "| asset | algorithm | threat class | horizon | risk | migratable |",
        "|---|---|---|---|---|---|",
    ]
    ordered = sorted(
        problem.assets, key=lambda a: -baseline.node_scores.get(a.name, 0.0)
    )
    for asset in ordered:
        horizon = _horizon_of(asset)
        migratable = (
            "yes"
            if asset.algorithm in problem.replacements and asset.controllable
            else "not yours"
            if asset.algorithm in problem.replacements
            else "no"
        )
        lines.append(
            f"| `{asset.name}` | {asset.algorithm} | "
            f"{asset.effective_threat_class.value} | {horizon} | "
            f"{baseline.node_scores.get(asset.name, 0.0):.1f} | {migratable} |"
        )
    lines.append("")
    return lines


def _horizon_of(asset) -> str:
    cls = asset.effective_threat_class
    if cls is ThreatClass.AUTHENTICATION and asset.credential_validity_years is not None:
        return f"{asset.credential_validity_years:.2f}y in service"
    if (
        cls is ThreatClass.NON_REPUDIATION
        and asset.verification_horizon_years is not None
    ):
        return f"{asset.verification_horizon_years:.0f}y verifiable"
    return f"{asset.data_lifetime_years:.0f}y retained"


def _footer() -> list[str]:
    return [
        "---",
        "",
        f"Generated by Q-SHIELD {__version__}. The model, its known failure modes "
        "and the experiments behind it are documented in `docs/METHODOLOGY.md`, "
        "`docs/FINDINGS.md` and `docs/IDENTITY.md`.",
    ]


def render_plan_only(decision: PlanningReport) -> str:
    """A one-line summary, for a CI step or a commit message."""
    chosen = ", ".join(decision.plan.selected) or "nothing"
    return (
        f"migrate {chosen} (cost {decision.plan.cost:.2f}, "
        f"objective {decision.plan.objective:.2f}, planner {decision.planner})"
    )
