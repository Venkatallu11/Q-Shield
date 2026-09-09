# Q-SHIELD

**A falsifiable model for attack-path-aware post-quantum migration planning.**

Q-SHIELD asks one question and tries hard to answer it in a way that could come
out "no":

> Does attack-path-aware, cost-constrained migration reduce modeled quantum
> exposure more efficiently than flat asset-priority ranking?

It is a research instrument, not a compliance tool. The objective is a scenario
model with uncalibrated parameters. A lower score means a plan is better *under
this model*; it does not mean a system is safer.

```bash
pip install -e ".[scan]"
qshield ingest --certs /etc/ssl/certs --budget 6 --output case.json
qshield report --input case.json --draws 1000 --output migration.md
```

---

## Three results worth knowing

**The one result checked against something other than itself.** The cause model
states a generative story, so it can be simulated. Doing that caught the model
shipped hours earlier: exact when a cause propagates with certainty, **biased
upward by nine standard errors** when propagation was partial and a cause stood
behind another cause. 0.10 enumerates cause states in topological order over the
DAG using direct parents only. On estates where two hardware modules share a
supplier, the independent-hops model overstates path risk by **13.8%**; where
services share one HSM, by **36.4%**. But mapping that supplier out changes the
risk *number* by 3 points and the *plan* on 1% of instances — a supplier is a
cause you cannot buy, so it moves the assessment, not the menu.

**"Independent hops" was a defect, not a caveat.** Two certificates issued by one
authority on the same attack path were raised to that authority's risk and then
multiplied together as independent events — but they are the *same* event. 0.9
conditions on shared causes instead of multiplying marginals. On estates where
services share an HSM or identity provider, the old model overstated path risk by
**36.4%**; where the two effects are decomposed, replacing max-dominance with a
noisy-OR marginal *raises* risk by 6-10 points while the correlation correction
*lowers* the path term by 18. Plans disagree on 63-79% of instances. It matters
exactly when two dependents of one cause lie on one path — the PKI topology never
produces that, so its correlation effect is precisely zero.


**Migration actions are complements, not substitutes — so greedy planning has no
approximation guarantee.** The risk-reduction function is *supermodular* on every
sampled pair, in both suites, with and without trust hierarchies: 0% supermodularity
violations against 38.8%/67.7% submodularity violations. The cause is the noisy-OR
path composition — the benefit of migrating one node is proportional to
`prod(1 - p)` over the others, which *grows* as they are migrated. Delegation is the
limiting case, where a leaf's benefit is exactly zero until its anchor moves. So the
`1 - 1/e` bound is unavailable and the 80.3% greedy-optimal rate stays an empirical
observation.

**Hybrid deployments rescue short-lived credentials.** Pure post-quantum only pays
off above a 3.7-year horizon; a TLS leaf certificate lives 90 days, so the
calibrated model said leave it alone. A hybrid is broken only if *both* halves are,
which moves the crossover to **0.15 years**. On a flat estate of short-lived service
credentials the pure policy migrates **nothing at all** across 300 instances, while
the hybrid ladder migrates 775 of 1643 credentials. On a certificate hierarchy it
changes almost nothing — trust inheritance had already excluded those leaves.

## Is it calibrated?

**Not against outcomes, and it cannot be.** No cryptographically relevant quantum
computer exists and nothing has ever been compromised via Shor's algorithm, so
there are zero outcome events to fit parameters against. Any tool advertising a
calibrated quantum risk score is claiming data nobody has.

What 0.7 does is replace invented constants with published estimates, and
separate the parameters that can be grounded from the ones that cannot. Doing
that found two errors in the model's own registry:

- **Elliptic curve is a far easier quantum target than RSA** at equal classical
  security — 2.6x fewer logical qubits, ~148x fewer Toffoli gates. Every version
  through 0.6 scored `RSA-2048`, `ECDSA` and `X25519` at exactly 1.00, asserting
  they fall at the same moment. An estate that sequenced RSA before ECDSA on this
  model sequenced them **backwards**.
- **Symmetric cryptography scored too high.** NIST makes AES-128 the *benchmark*
  for post-quantum security, yet the registry rated `AES-128` (0.15) and
  `AES-256` (0.08) as riskier than `ML-KEM` (0.05) — more quantum risk than the
  algorithms it recommends migrating to.

And it produced a constraint the old model could not express: **migration is not
always worth doing.** Below a crossover horizon, the chance a CRQC arrives at all
is smaller than the residual risk of a young post-quantum primitive. `ECDSA →
ML-DSA` pays off only above **3.7 years**; a 90-day TLS certificate is 0.25.

Honest headline: calibration corrected two real errors and mostly *confirmed* the
existing recommendations — 93.3% plan agreement on PKI instances, and identical
plans on both real-shaped estates. See **[docs/CALIBRATION.md](docs/CALIBRATION.md)**.

## Can you act on it without calibrating it?

A certificate supplies about half of what the model needs — algorithm, validity
period, issuing authority — and none of the rest. Nothing in X.509 records how
sensitive the data behind a key is, how exposed the host is, or what migrating it
would cost.

Rather than guess and present the result as a measurement, `qshield robustness`
resamples every unobservable parameter across its full plausible range, replans
each time, and reports what survives. Over 1000 redraws, on a constructed CA
hierarchy and on a real certificate estate ingested from PEM files:

| tier | affordable in | chosen when affordable |
|---|---|---|
| root CA | 28.7% / 34.8% of draws | **91.3% / 83.3%** |
| intermediate CA | 76.1% / 100% | 21.8% / 34.3% |
| TLS leaves | **100%** of draws | **0.8% – 3.5%** |

Leaf certificates are affordable in every draw and almost never worth migrating;
the root is chosen nearly every time it can be paid for. **That recommendation
does not depend on any guessed parameter** — it holds across their entire
admissible range, on both estates. So it can be acted on today, without
calibrating anything.

On both estates the root came back *blocked by budget*: the binding constraint
was money, not analysis.

See **[docs/USING.md](docs/USING.md)**.

---

## What the experiments found

Full detail and reproduction seeds in **[docs/FINDINGS.md](docs/FINDINGS.md)**;
identity infrastructure in **[docs/IDENTITY.md](docs/IDENTITY.md)**.

### Identity: a migration plan that buys nothing

> **Migrating leaf certificates while their issuer still signs with a classical
> algorithm reduces risk by exactly zero.**

A forged root certificate forges everything beneath it — immediately, with
nothing in between to defeat. So a leaf's effective risk is pinned to its
issuer's, and a planner that cannot see trust relations spends its budget on
cheap, highly-exposed leaves. Over 300 synthetic PKI instances:

- trust-blind planning wastes **88.8%** of its budget, and in **265 of 300**
  instances the *entire* spend achieves nothing;
- a flat-priority heuristic wastes **99.4%**, with **297 of 300** spends
  achieving nothing at all;
- the blind planners *report more improvement* than the correct model — 2.2
  against 0.57 — while delivering a quarter of it;
- trust-aware and trust-blind planning agree on **0 of 300** instances.

Bounded honestly: this needs trust anchors costing more than ~2.5x a leaf. Below
that a blind planner buys the anchor anyway and the refinement barely matters.

Separately, harvest-now-decrypt-later **does not apply to signatures**. Recording
a signature gains an adversary nothing; forgery starts when a CRQC exists, not
retroactively. Scoring signing keys by data retention — which every version
through 0.4 did — made a 90-day TLS leaf and the 20-year root above it score
*identically*. Correcting it inverts the recommendation: **1153 leaves / 33 roots**
selected under the old scoring, **8 leaves / 175 roots** under the corrected one,
with plans agreeing on 2.3% of instances.

For an individual, ~70% of modeled quantum risk sits on assets they do not
control, and the residual concentrates 2.3:1 in the class where waiting is
irreversible. Q-SHIELD models none of phishing, SIM-swap or credential theft,
which dominate real identity compromise — see the scope note in IDENTITY.md.

### The generic model

- **Path-awareness helps, modestly, on the objective.** Pooled over 300 instances
  and all 26 distinct weightings: mean improvement **+0.758** (95% CI
  [0.719, 0.796]) over an equally-powerful graph-blind planner, favourable at
  **26 of 26** weightings. The effect survives weight misspecification.
- **But it changes nothing 69% of the time.** Path-aware and node-only planners
  pick the identical plan on roughly two thirds of instances. Where the plans
  differ, the gap averages ~2.5 points on a 0-100 scale.
- **And it makes the worst case worse.** On worst-case path exposure — the one
  metric no planner optimises — path-awareness *loses*: mean **−0.736**
  (95% CI [−0.971, −0.516]), losing on 86 of 125 decisive instances. Minimising
  *average* path risk trades away the *maximum*. If you care about the worst way
  into your system, this objective is the wrong one as it stands.
- **Exhaustive search is mostly unnecessary.** A greedy cost-benefit planner
  recovers the exact optimum on **80.3%** of instances at a mean optimality gap of
  0.248, without the `O(2^n)` enumeration.
- **The worst-case failure is fixable, at a price.** Replacing the mean with a
  conditional value at risk over the worst 15% of paths turns a −0.736 worst-case
  deficit into a **+0.171** advantage — but costs six sevenths of the mean-case
  benefit (+2.487 → +0.412). A frontier, not a free upgrade, so the tail-aware
  setting is opt-in.

## What the previous version got wrong

Q-SHIELD 0.3 reported `wins 483, losses 0, ties 517` over 1000 instances as
evidence for the hypothesis. Zero losses was not a strong result — it was
**structurally impossible to lose**. The heuristic returns a budget-feasible
subset; the optimizer minimises over *all* budget-feasible subsets; so the
heuristic's answer was always already inside the optimizer's search space. The
comparison measured subset enumeration, not migration strategy, and could not have
been falsified.

Five further defects, each now pinned by a test in
[`tests/test_regressions.py`](tests/test_regressions.py):

| # | defect | consequence |
|---|---|---|
| R1 | rotation-pressure term averaged only over assets scoring above 10 | migrating RSA-2048 to ML-KEM *raised* the objective, 18.838 → 19.329 |
| R2 | `path_risk` read only the target of each hop | the entrypoint's own crypto never affected the path term |
| R3 | two divergent algorithm taxonomies | `RSA`, `Kyber`, `Dilithium`, `SHA-256` scored as "unknown" on one code path |
| R4 | weight grid not de-duplicated on the simplex | two of 27 sweep points were the same weighting |
| R5 | two path enumerators with off-by-one depth guards | the same graph gave different path sets per module |
| R6 | optimizer enumerated from `r=1` | "migrate nothing" was not a candidate plan |
| R9 | the shipped optimizer ignored objective weights | the weight sweep re-implemented it inline, and the copies drifted |
| R10 | asymmetric percentile indices (`int(.05n)-1` vs `int(.95n)`) | every reported interval was biased |

Plus the statistical error that mattered most: the three Monte Carlo runs used
**different seeds per strategy** (11, 12, 13), so the percentile bands printed
side by side were computed on different random draws and were not comparable.
This version uses common random numbers throughout.

## Install

```bash
pip install -e ".[scan]"   # adds cryptography, for reading certificates
pip install -e ".[dev]"    # tests and linting
```

Python 3.10+. The model and every experiment have **no runtime dependencies**, so
no reported number can shift because a third-party default changed. Only the
certificate parser takes one, and `qshield` imports cleanly without it — there is
a test that asserts exactly this.

## Use

```bash
# Real infrastructure in, decision out
qshield ingest --certs /etc/ssl/certs ./pki --budget 6 --output case.json
qshield ingest --tls api.example.com www.example.com --budget 6 --output case.json
qshield robustness --input case.json --draws 1000
qshield report --input case.json --draws 1000 --output migration.md
```

```bash
# Full report for one system, with paired uncertainty analysis
qshield benchmark --input cases/chokepoint_case.json --trials 4000 \
                  --output results/benchmark.json

# Is the benefit path-awareness, or just exhaustive search?
qshield ablation --instances 400 --trials 200 --output results/ablation.json

# Do the plans survive being wrong about the weights?
qshield generalization --instances 300 --output results/generalization.json

# Is the recommendation stable across the weight simplex?
qshield sensitivity --input cases/reference_case.json

# Identity: how much of a trust-blind plan buys nothing?
qshield hierarchy --instances 300 --output results/hierarchy.json
qshield hierarchy --instances 200 --sweep-root-cost   # bound the claim

# Does scoring signatures by credential validity change the plan?
qshield threat-class --instances 300

# Does a tail-aware path term repair the worst-case regression?
qshield tail-risk --instances 400

# Personal identity: how much of the risk is yours to fix?
qshield personal --instances 300

# CycloneDX cryptographic bill of materials
qshield cbom --input cases/reference_case.json
```

Reproduce everything in `results/` with `bash scripts/run_all.sh` (~2.5 minutes).

```python
from qshield import AssetModel, EdgeModel, MigrationProblem, exhaustive, pareto_frontier

problem = MigrationProblem(
    assets=(AssetModel("api", "RSA-2048", sensitivity=0.9, exposure=0.95,
                       data_lifetime_years=15, dependency_count=8,
                       rotation_hours=24, migration_cost=1.5),
            AssetModel("db", "AES-256", sensitivity=1.0, exposure=0.4,
                       data_lifetime_years=15, migration_cost=0.5)),
    edges=(EdgeModel("api", "db", reliability=0.9),),
    entrypoints=("api",), targets=("db",),
    replacements={"RSA-2048": "ML-KEM"}, budget=2.0,
)
best, all_plans = exhaustive(problem)
print(best.as_dict())
print([p.as_dict() for p in pareto_frontier(all_plans)])
```

## How it works

```
qshield/
  algorithms.py     single algorithm registry: aliases, families, primitive kinds
  threat.py         threat classes and the horizon each is actually exposed over
  paths.py          attack-path enumeration + delegation (trust) edges
  model.py          the objective -- node risk, trust inheritance, CVaR path risk
  optimizer.py      exhaustive / greedy planners, Pareto frontier
  planner.py        picks a planner that will finish, and says when it downgraded
  calibration.py    published resource estimates, CRQC forecasts, Mosca's rule
  hybrid.py         classical+PQC compositions and the migration ladder
  correlation.py    shared causes; path risk as a joint, not a product
  uncertainty.py    common-random-number sampling, paired stats, bootstrap CIs
  ingest/           real artifacts in: X.509 parsing, TLS scanning, provenance
  report.py         the Markdown a person actually reads
  cbom.py           CycloneDX export
  cli.py            qshield <command>
  experiments/      generator, pki, benchmark, ablation, generalization,
                    sensitivity, tail_risk, hierarchy, threat_class, personal,
                    robustness, calibration_impact, curvature, hybrid_ladder,
                    correlation_impact
```

The objective is a weighted mean of node risk, path risk and key-rotation
pressure. Its central property is **monotonicity**: replacing an algorithm with a
strictly less susceptible one can never increase the score. 0.3 violated this;
`tests/test_monotonicity.py` now tests it by construction, over every feasible
plan of randomised instances, and under randomised weightings.

The experimental design uses three arms so that path-awareness and search power
can be told apart, and scores them on metrics no arm optimises so that any arm
can lose. See **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

## Limits

- **Not validated against reality.** Quantum factors and objective weights encode
  an ordering hypothesis; they are not measured. Calibration against real
  inventories and incident data is the missing step.
- **Synthetic instances.** Every quantitative result comes from the generator in
  `experiments/generator.py`, whose structure is itself a hypothesis about what
  real infrastructure looks like.
- **Cause edges fire independently.** Causes over causes are modelled as of 0.10,
  but a defect that deterministically affects every unit of one firmware build is
  not the same as two independent firings, and is not represented.
- **Exponential exact planner.** `exhaustive` is `O(2^n)` in migration candidates
  and exists as a reference. Use `greedy_marginal` beyond ~20 candidates.
- **Quantum risk only.** Phishing, credential stuffing, SIM-swap, session-token
  theft, malware and device theft are not modelled and dominate real identity
  compromise. Nothing here is a security assessment.
- **The identity results are error sizes, not win rates.** The corrected planner
  minimises the metric it is scored on, so it cannot lose; what is measured is how
  large the modelling error is, and in the cheap-anchor regime it is nearly zero.
- **Ingested parameters are half assumptions.** A certificate cannot tell you
  sensitivity, exposure, data lifetime, criticality, cost or rotation cadence.
  Every generated case records which fields were observed and which were filled
  in; run the robustness pass before believing anything that depends on them.
- **No standard is implemented and no conformance is claimed.**

## Tests

```bash
pytest                  # 393 tests
pytest -m "not slow"    # skip the full-experiment smoke tests
```

## Documentation

| | |
|---|---|
| [docs/USING.md](docs/USING.md) | running it on a real estate |
| [docs/CALIBRATION.md](docs/CALIBRATION.md) | what is grounded in published data, and what is not |
| [docs/IDENTITY.md](docs/IDENTITY.md) | certificate hierarchies and the crypto-agility trap |
| [docs/FINDINGS.md](docs/FINDINGS.md) | every measured result, with reproduction seeds |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | the model, and why it is shaped this way |

## License

Apache-2.0. See [LICENSE](LICENSE).
