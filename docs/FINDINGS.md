# Findings

All numbers below were produced by `scripts/run_all.sh` at the commit that added
this file, and are reproducible from the seeds recorded inside each report in
`results/`. Effects are paired per instance and intervals are 95% percentile
bootstrap.

Read the [caveats](#what-none-of-this-establishes) before quoting anything here.

---

## 1. The 0.3 headline result was not evidence

Q-SHIELD 0.3 reported, over 1000 randomised instances:

```
wins 483   losses 0   ties 517   win_rate 0.483
```

Zero losses in 1000 trials is not a strong result; it is a sign that losing was
impossible. The flat-priority heuristic returns a budget-feasible subset of the
migration candidates. The optimizer minimises the objective over *every*
budget-feasible subset. The heuristic's answer is therefore one of the candidates
the optimizer already evaluated, so

```
objective(optimizer) <= objective(heuristic)
```

holds for every instance, every graph and every parameter setting. It is a
property of subset enumeration, not a claim about migration strategy that data
could have contradicted.

This is asserted directly in `tests/test_optimizer.py::
test_heuristic_plans_lie_inside_the_exhaustive_search_space` and in
`tests/test_regressions.py::test_r7_...`, so the claim cannot be revived by
accident.

The design also **confounded two effects**. "Exhaustive search beats greedy" and
"modelling attack paths beats ranking assets" are different claims, and 0.3's
two-arm comparison cannot separate them. Everything below uses a third arm to do
so.

## 2. Design: three arms, two of them controls

| arm | search | sees the graph | isolates |
|---|---|---|---|
| `flat_priority` | greedy | no | neither effect |
| `node_only` | exhaustive | no | search power alone |
| `path_aware` | exhaustive | yes | search power + path-awareness |

`node_only` minimises the same objective with the path weight set to zero. It has
identical search power to `path_aware`, so the `path_aware - node_only` contrast
is the effect of path-awareness alone.

Arms are scored on metrics none of them minimises, so any arm can lose:

- **`path_max`** — the worst single attack path. Fully independent: no arm
  targets it.
- **`expected_objective`** — mean objective under parameter uncertainty. Only
  *weakly* held out: it is the expectation of the same function `path_aware`
  minimises, and the results below confirm it behaves almost identically to the
  nominal objective. It is reported for completeness and carries little
  independent weight.

400 instances, 200 Monte Carlo trials each, common random numbers across arms.

## 3. Path-awareness helps on the objective — and the effect is robust

Contrast `node_only - path_aware`, positive means path-aware scored lower:

| metric | mean Δ | 95% CI | ties | decisive | path-aware wins when decisive |
|---|---|---|---|---|---|
| `objective` (in-objective) | +0.543 | [0.432, 0.659] | 68.8% | 125 | 100% *(guaranteed)* |
| `expected_objective` | +0.515 | [0.409, 0.624] | 68.8% | 125 | 99.2% |

The generalization experiment (300 instances scored under all 26 distinct
weightings on the simplex, plans chosen under the default weights) is the
stronger test, because under mismatched weights the planner is minimising a
different function than it is scored on and **has no guarantee at all**:

```
pooled node_only - path_aware:  mean +0.758  CI [0.719, 0.796]
                                7800 comparisons, 69.7% ties
                                path_aware better 2289, node_only better 77
                                when decisive: +2.499  CI [2.403, 2.595]
robustness: favourable at 26 of 26 weightings, adverse at 0
```

77 genuine losses confirm the test could have failed. It did not: the advantage
holds across the entire weight simplex, so it is not an artifact of the
particular uncalibrated weights the model ships with.

**Effect size in context.** Two thirds of instances tie — path-aware and
node-only pick the *same plan* on 68.8% of instances, so path-awareness changes
nothing at all most of the time. Where the plans differ, the objective gap
averages ~2.5 points on a 0-100 scale. The honest summary is a modest effect on
roughly a third of instances, not a general improvement.

## 4. Negative result: path-awareness makes the worst case worse

On `path_max`, the one fully held-out metric, the sign reverses:

| contrast | mean Δ | 95% CI | path-aware wins when decisive |
|---|---|---|---|
| `node_only - path_aware` | **−0.736** | **[−0.971, −0.516]** | 31.2% (39 of 125) |
| `flat_priority - path_aware` | −0.227 | [−0.503, +0.039] | 49.3% (72 of 146) |

The interval on the first row excludes zero, and path-aware loses on 86 of 125
decisive instances. **Optimising mean path risk systematically degrades
worst-case path risk.**

The mechanism is not mysterious: the objective averages over paths, and the
cheapest way to reduce a mean is to improve the many moderate paths rather than
the single worst one. A planner told to minimise average exposure will trade away
the maximum to do it.

This matters operationally. An analyst who cares about the worst way into their
system — which is the usual threat-modelling posture — should not use this
objective as it stands. Either weight `path_max` into the objective directly, or
plan against a max-based objective. The current model does neither, and the
`path_max` column in every report exists so that this stays visible.

This is the single most useful result in the suite, and 0.3's design could not
have produced it: with only an in-objective comparison there was no metric on
which the method could be seen to fail.

## 5. Search power is a smaller, separate effect

Contrast `flat_priority - node_only`, isolating exhaustive search with both arms
graph-blind:

| metric | mean Δ | 95% CI | node-only wins when decisive |
|---|---|---|---|
| `objective` | +0.202 | [0.093, 0.317] | 62.6% |
| `path_max` | +0.509 | [0.330, 0.699] | 73.6% |

Exhaustive search is worth roughly a third of what path-awareness is worth on the
objective — and unlike path-awareness it *improves* worst-case exposure. The two
effects are real and separable, which is precisely what 0.3's two-arm design
could not show.

## 6. Exponential search is mostly unnecessary

`greedy_marginal` (repeatedly take the best objective reduction per unit cost)
against the exhaustive optimum over 300 instances (`scripts/greedy_gap.py`,
`results/greedy_gap.json`):

```
identical plan chosen:   241 / 300  (80.3%)
mean optimality gap:     0.248        p95: 1.659        max: 4.813
```

The exhaustive planner is `O(2^n)` in migration candidates and exists as an exact
reference. For instances beyond roughly 20 candidates, the greedy planner
recovers the same plan four times out of five at a mean cost of a quarter of an
objective point. Nothing in the suite justifies the exponential method
operationally.

## 7. Plan recommendations are stable under weight uncertainty

Both shipped cases were re-optimised under all 26 distinct weightings:

| case | distinct plans | modal plan share | max regret vs per-weighting optimum |
|---|---|---|---|
| `reference_case` | 1 of 26 | 100% | 0.0 |
| `chokepoint_case` | 1 of 26 | 100% | 0.0 |

On these two cases the recommendation does not move anywhere on the simplex. The
0.3 README's concern that uncalibrated weights undermine the output is, at least
here, not borne out — the weights are uncertain but the *decision* is not
sensitive to them. This does not generalise; run `qshield sensitivity` on your own
case rather than assuming it.

## 8. The reference case is not diagnostic

On `cases/reference_case.json`, every arm — `flat_priority`, `greedy_marginal`
and `path_aware` — selects the same plan (`identity`, `public-api`, cost 2.5,
objective 11.286). The case 0.3 shipped as its benchmark cannot distinguish the
method it was meant to demonstrate.

`cases/chokepoint_case.json` is a hand-built illustration that does separate them:

| arm | selects | objective |
|---|---|---|
| `do_nothing` | — | 15.729 |
| `flat_priority` | `archive-signing` | 12.136 |
| `node_only` | `archive-signing` | 12.136 |
| `greedy_marginal` | `service-mesh-ca` | 7.401 |
| `path_aware` | `service-mesh-ca` | 7.401 |

`service-mesh-ca` scores *lower* on standalone node risk (24.0) than
`archive-signing` (30.0), so both graph-blind arms buy the archive. But it sits on
the only path from the entrypoint to the target, and the archive sits on none.
This case is constructed to make the mechanism visible; it demonstrates that the
mechanism exists, **not** how often it matters. Section 3 answers that.

## 10. The worst-case regression is fixable, at a price (0.5)

Section 4 found that minimising the *mean* of the path risks made worst-case path
exposure significantly worse. 0.5 replaces the mean with a conditional value at
risk over the worst `alpha` fraction of paths and sweeps `alpha`
(`results/tail_risk.json`, 400 instances, control = exhaustive search with no
path term at all):

| `path_alpha` | paths averaged | worst-case Δ | 95% CI | mean-path Δ | 95% CI |
|---|---|---|---|---|---|
| 1.00 *(0.4 behaviour)* | 4.40 | **−0.736** | [−0.971, −0.516] | +2.487 | [+2.017, +2.970] |
| 0.75 | 3.65 | −0.506 | [−0.707, −0.319] | +2.278 | [+1.821, +2.755] |
| 0.50 | 2.42 | −0.157 | [−0.307, −0.011] | +1.415 | [+1.031, +1.819] |
| 0.35 | 2.13 | −0.060 | [−0.188, +0.064] | +1.197 | [+0.851, +1.566] |
| 0.25 | 1.43 | +0.079 | [−0.017, +0.177] | +0.714 | [+0.413, +1.037] |
| **0.15** | 1.16 | **+0.171** | [+0.096, +0.253] | +0.412 | [+0.180, +0.667] |
| 0.05 | 1.00 | +0.191 | [+0.117, +0.272] | +0.396 | [+0.175, +0.651] |

Positive means better than the path-blind control. Only `alpha <= 0.15` beats it
on worst-case **and** mean with intervals excluding zero, so that is what
`TAIL_AWARE_WEIGHTS` ships as. An earlier pilot at 80 instances suggested 0.25
sufficed; at 400 instances its interval straddles zero, which is why the default
is set from the full run and not the pilot.

**The fix is not free.** Moving from `alpha = 1` to `alpha = 0.15` converts a
−0.736 worst-case deficit into a +0.171 advantage, but the mean-path advantage
falls from +2.487 to +0.412 — six sevenths of it. This is a genuine frontier, not
a strictly better setting, and the right point on it depends on whether you are
defending against the average way in or the worst one. `DEFAULT_WEIGHTS` still
ships `alpha = 1.0` so that existing results remain comparable; the tail-aware
setting is opt-in and documented rather than silently substituted.

## 11. Identity infrastructure (0.5)

Two errors that specifically break certificate hierarchies, and the operational
consequence — a migration plan that spends real budget for **exactly zero** risk
reduction — are documented with their measurements in
**[IDENTITY.md](IDENTITY.md)**. Headline numbers, over 300 synthetic PKI
instances:

- Scoring signatures by data-retention rather than credential-validity horizon
  inverts the recommendation: 1153 leaves and 33 roots selected under the old
  scoring, 8 leaves and 175 roots under the corrected one. Plans agree on 2.3% of
  instances; real risk reduction 1.205 versus 4.290.
- Trust-blind planning wastes **88.8%** of its budget, and in 265 of 300
  instances the *entire* spend achieves nothing. A flat-priority heuristic wastes
  99.4%, with 297 of 300 spends achieving nothing at all.
- Blind planners report *higher* improvement than the correct model — 2.2 against
  0.57 — while delivering a quarter of it.
- The effect is bounded: it requires trust anchors costing more than ~2.5x a leaf.
  Below that a blind planner buys the anchor anyway. See the root-cost sweep.

For an individual, ~70% of modeled quantum risk sits on assets they do not
control, a realistic budget removes ~12% of the objective, and the residual is
concentrated 2.3:1 in the confidentiality class — the one where waiting is
irreversible. Q-SHIELD models none of phishing, SIM-swap or credential theft,
which dominate real personal identity compromise.

## What none of this establishes

- **Nothing here is validated against reality.** Quantum factors and objective
  weights are scenario parameters chosen to encode an ordering hypothesis. A
  lower objective means a plan scores better *under this model*. It does not mean
  a system is safer. Calibration against real inventories and incident data is
  the missing step, and no amount of internal consistency substitutes for it.
- **The instances are synthetic.** Every quantitative result above comes from the
  generator in `qshield/experiments/generator.py`. Its structure — a guaranteed
  backbone, random cross-links, cost correlated with dependency count — is itself
  a hypothesis about what real infrastructure looks like. Results may not transfer
  to real topologies.
- **Path risk assumes independent hops.** Real infrastructure shares platforms,
  libraries and operators, so compromises correlate. This will overstate the
  benefit of breaking any single chain.
- **The objective is a scenario score, not a probability.** Node scores are read
  as probabilities inside `path_risk` purely as a combination rule.
- **Effects are small and mostly absent.** Two thirds of instances tie. Quoting
  the conditional effect without the tie rate would misrepresent the result.
- **The identity results are not win rates and must not be quoted as such.** In
  sections 10 and 11 the corrected planner minimises the metric it is scored on
  and every rival plan is feasible for it, so it cannot lose. What is reported is
  the size of a modelling error, which was free to come out zero and in the
  cheap-anchor regime very nearly does.
- **The trust semantics are a modelling choice.** That a forged root certificate
  forges everything beneath it is a cryptographic fact; that this is best modelled
  as complete max-dominance is a decision, and the magnitude of every number in
  section 11 depends on it along with the generator's cost structure.
- **Q-SHIELD models quantum risk only.** For identity in particular this excludes
  phishing, credential stuffing, SIM-swap, session-token theft, malware and device
  theft, which are the dominant real-world threats. Nothing here is a security
  assessment.
