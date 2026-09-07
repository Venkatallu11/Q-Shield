"""Command line entry point: ``qshield <experiment> [options]``.

0.3 shipped four single-purpose scripts with hard-coded seeds, case paths and
output filenames baked into the source, so reproducing a run with different
parameters meant editing the script. Every knob here is a flag, and every report
records the flags it was produced with.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .model import DEFAULT_WEIGHTS, ObjectiveWeights
from .uncertainty import PerturbationModel

DEFAULT_SEED = 20260907


def _weights(raw: str | None) -> ObjectiveWeights:
    """Parse ``node,path,rotation``.

    Raises :class:`argparse.ArgumentTypeError` so that :func:`main` can convert it
    into the usage message and exit code argparse would produce for any other bad
    argument, rather than a traceback.
    """
    if not raw:
        return DEFAULT_WEIGHTS
    try:
        parts = [float(x) for x in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--weights expects numbers: {exc}") from exc
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--weights expects node,path,rotation (got {len(parts)} values)"
        )
    try:
        return ObjectiveWeights(*parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--weights: {exc}") from exc


def _ingest(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Assemble a case from certificates on disk and/or live TLS endpoints."""
    if not args.certs and not args.tls:
        parser.error("ingest needs --certs and/or --tls")
    try:
        from .ingest.x509 import fetch_tls_chain, from_certificates, load_certificates
    except ImportError as exc:
        parser.error(str(exc))

    collected = []
    if args.certs:
        collected.extend(load_certificates(args.certs))
    for endpoint in args.tls:
        host, _, port = endpoint.partition(":")
        try:
            chain = fetch_tls_chain(host, int(port) if port else 443)
        except OSError as exc:
            # One unreachable host must not discard the rest of a scan.
            print(f"warning: {endpoint}: {exc}", file=sys.stderr)
            continue
        collected.extend((cert, endpoint) for cert in chain)

    if not collected:
        parser.error("no certificates were read")

    inventory = from_certificates(collected)
    return inventory.as_case(
        budget=args.budget,
        entrypoints=args.entrypoints,
        targets=args.targets,
    )


def _report(args: argparse.Namespace, weights: ObjectiveWeights) -> int:
    """Render Markdown rather than JSON, so this path bypasses _emit."""
    import json as _json

    from .experiments.benchmark import load_case
    from .report import render

    problem = load_case(args.input)
    robustness = None
    if args.draws > 0:
        from .experiments.robustness import run as run_robustness

        robustness = run_robustness(
            problem, draws=args.draws, seed=args.seed, weights=weights
        )
    provenance = _json.loads(Path(args.input).read_text(encoding="utf-8")).get(
        "_provenance"
    )
    text = render(
        problem,
        title=args.title,
        weights=weights,
        robustness=robustness,
        provenance=provenance,
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    if not args.quiet:
        print(text)
    return 0


def _emit(report: dict[str, Any], output: str | None, quiet: bool) -> None:
    text = json.dumps(report, indent=2, default=str)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {output}", file=sys.stderr)
    if not quiet:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qshield",
        description="Falsifiable experiments on post-quantum migration planning.",
    )
    parser.add_argument("--version", action="version", version=f"qshield {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=DEFAULT_SEED)
    common.add_argument("--output", help="write the JSON report here")
    common.add_argument("--quiet", action="store_true", help="do not echo the report")
    common.add_argument("--weights", type=str, default=None, metavar="N,P,R")

    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark", parents=[common], help="full report for one case file")
    b.add_argument("--input", required=True)
    b.add_argument("--trials", type=int, default=3000)

    a = sub.add_parser(
        "ablation",
        parents=[common],
        help="separate path-awareness from search power on held-out metrics",
    )
    a.add_argument("--instances", type=int, default=300)
    a.add_argument("--trials", type=int, default=150)

    g = sub.add_parser(
        "generalization",
        parents=[common],
        help="score plans under weights they were not chosen with",
    )
    g.add_argument("--instances", type=int, default=300)

    s = sub.add_parser(
        "sensitivity", parents=[common], help="plan stability across the weight simplex"
    )
    s.add_argument("--input", required=True)

    t = sub.add_parser(
        "tail-risk",
        parents=[common],
        help="sweep the path tail parameter against the worst-case regression",
    )
    t.add_argument("--instances", type=int, default=300)

    h = sub.add_parser(
        "hierarchy",
        parents=[common],
        help="the crypto-agility trap: how much of a trust-blind plan buys nothing?",
    )
    h.add_argument("--instances", type=int, default=300)
    h.add_argument(
        "--sweep-root-cost",
        action="store_true",
        help="vary the trust anchor's migration cost to bound the effect",
    )

    tc = sub.add_parser(
        "threat-class",
        parents=[common],
        help="does scoring signatures against a credential horizon change the plan?",
    )
    tc.add_argument("--instances", type=int, default=300)

    pi = sub.add_parser(
        "personal",
        parents=[common],
        help="personal identity: how much of the risk is the individual's to fix?",
    )
    pi.add_argument("--instances", type=int, default=300)

    ing = sub.add_parser(
        "ingest",
        parents=[common],
        help="build a case from real certificates (files, directories, or live TLS)",
    )
    ing.add_argument(
        "--certs",
        nargs="+",
        default=[],
        metavar="PATH",
        help="PEM/DER files or directories to scan",
    )
    ing.add_argument(
        "--tls",
        nargs="+",
        default=[],
        metavar="HOST[:PORT]",
        help="TLS endpoints to fetch chains from (port defaults to 443)",
    )
    ing.add_argument("--budget", type=float, default=3.0)
    ing.add_argument("--entrypoints", nargs="*", default=[])
    ing.add_argument("--targets", nargs="*", default=[])

    rob = sub.add_parser(
        "robustness",
        parents=[common],
        help="which recommendations survive the parameters that had to be guessed?",
    )
    rob.add_argument("--input", required=True)
    rob.add_argument("--draws", type=int, default=500)

    rep = sub.add_parser(
        "report", parents=[common], help="render a migration report as Markdown"
    )
    rep.add_argument("--input", required=True)
    rep.add_argument("--draws", type=int, default=500, help="0 skips the robustness pass")
    rep.add_argument("--title", default="Post-quantum migration report")

    c = sub.add_parser("cbom", parents=[common], help="export a CycloneDX CBOM")
    c.add_argument("--input", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        weights = _weights(args.weights)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))  # exits with status 2
    perturbation = PerturbationModel(quantum_factor_spread=0.25, weight_spread=0.20)

    if args.command == "benchmark":
        from .experiments.benchmark import load_case, run

        report = run(
            load_case(args.input),
            trials=args.trials,
            seed=args.seed,
            weights=weights,
            perturbation=perturbation,
        )
    elif args.command == "ablation":
        from .experiments.ablation import run

        report = run(
            instances=args.instances,
            trials=args.trials,
            seed=args.seed,
            weights=weights,
            perturbation=perturbation,
        )
    elif args.command == "generalization":
        from .experiments.generalization import run

        report = run(instances=args.instances, seed=args.seed, train_weights=weights)
    elif args.command == "sensitivity":
        from .experiments.benchmark import load_case
        from .experiments.sensitivity import run

        report = run(load_case(args.input), reference=weights)
    elif args.command == "tail-risk":
        from .experiments.tail_risk import run

        report = run(instances=args.instances, seed=args.seed, base_weights=weights)
    elif args.command == "hierarchy":
        from .experiments.hierarchy import run, sweep_root_cost

        report = (
            sweep_root_cost(instances=args.instances, seed=args.seed, weights=weights)
            if args.sweep_root_cost
            else run(instances=args.instances, seed=args.seed, weights=weights)
        )
    elif args.command == "threat-class":
        from .experiments.threat_class import run

        report = run(instances=args.instances, seed=args.seed, weights=weights)
    elif args.command == "personal":
        from .experiments.personal import run

        report = run(instances=args.instances, seed=args.seed, weights=weights)
    elif args.command == "ingest":
        report = _ingest(args, parser)
    elif args.command == "robustness":
        from .experiments.benchmark import load_case
        from .experiments.robustness import run

        report = run(load_case(args.input), draws=args.draws, seed=args.seed,
                     weights=weights)
    elif args.command == "report":
        return _report(args, weights)
    elif args.command == "cbom":
        from .cbom import to_cbom
        from .experiments.benchmark import load_case

        report = to_cbom(load_case(args.input).assets)
    else:  # pragma: no cover - argparse enforces the choices
        raise SystemExit(f"unknown command {args.command}")

    _emit(report, args.output, args.quiet)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
