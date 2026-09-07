# Methodology

## The objective

Q-SHIELD scores a system in `[0, 100]`, lower being better, as a weighted mean of
three components:

```
objective = (w_node . node_mean + w_path . path_mean + w_rot . rotation_pressure)
            / (w_node + w_path + w_rot)
```

Because the expression is normalised by the weight sum, only the *ratios* matter:
`(0.2, 0.2, 0.1)` and `(0.6, 0.6, 0.3)` are the same weighting.
`ObjectiveWeights.key()` exposes that identity so weight sweeps can de-duplicate
on the simplex.

### Node risk

```
node_risk(a) = 100 . exposure . sensitivity . longevity . quantum_factor . dependency
longevity  = clamp(data_lifetime_years / 20, 0.1, 1.0)
dependency = min(1.0, 0.5 + 0.1 . (dependency_count - 1))
```

Every factor lies in `(0, 1]`, so the product is monotone non-decreasing in each.
The 20-year longevity horizon encodes "harvest now, decrypt later": data that must
stay confidential past a plausible CRQC arrival is already fully exposed.

### Path risk

For a path `n0 -> n1 -> ... -> nk`, each node score is read as a bounded scenario
probability `p = clamp(score/100, 0, 0.99)` and the chain is treated as surviving
only if every step fails:

```
survive = (1 - p(n0)) . PROD over hops (u,v) of (1 - p(v) . reliability(u,v))
path_risk = 100 . (1 - survive)
```

Two properties matter. The entrypoint's own score is included — 0.3 omitted it,
so migrating an internet-facing entrypoint left the path term unchanged. And the
expression is monotone non-decreasing in every node score, since `1 - PROD(1 - x)`
increases in each `x`.

Hops are assumed independent. Real infrastructure shares platforms, libraries and
operators, so this overstates the value of breaking any single chain.

### Rotation pressure

```
rotation_i = node_score(i) . min(1, rotation_hours(i) / 168)
```

averaged over assets with `rotation_hours > 0`. Membership depends only on
`rotation_hours`, which migration never changes.

0.3 instead averaged over `{a : rotation_hours and score(a) > 10}`. Because
membership depended on the score, migrating an asset could drop it out of the
average and raise the mean over the survivors — making a strictly beneficial
migration *increase* the objective. See [Monotonicity](#monotonicity).

## Monotonicity

> Replacing an algorithm with a strictly less susceptible one can never increase
> the objective.

Each component is monotone non-decreasing in every node score, and every node
score is monotone non-decreasing in its quantum factor, so the weighted mean is
too. This is what makes the planner well-behaved: superset plans dominate subset
plans, and the null plan is never strictly better than migrating.

0.3 violated it. The counterexample is pinned in
`tests/test_regressions.py::test_r1_migration_cannot_increase_the_objective`:
two assets, migrating one from RSA-2048 to ML-KEM moved the objective from
18.838393 to **19.329375**. `tests/test_monotonicity.py` tests the property by
construction, across every feasible plan of randomised instances, and under
randomised weightings.

## Planners

| planner | method | guarantee |
|---|---|---|
| `do_nothing` | null plan | the reference for `improvement` |
| `greedy_flat` | rank by `exposure . sensitivity . lifetime`, first-fit | graph-blind by design |
| `greedy_marginal` | best objective reduction per unit cost, repeatedly | none |
| `exhaustive` | minimise over all budget-feasible subsets | exact, `O(2^n)` |

`exhaustive` breaks ties by cost, then lexicographically, so results do not depend
on enumeration order.

### The structural guarantee, and why it makes in-objective comparison useless

Every heuristic here returns a budget-feasible subset. `exhaustive` minimises over
all budget-feasible subsets. So for any heuristic `h`:

```
objective(exhaustive) <= objective(h)
```

always. Comparing them on that objective measures subset enumeration, not
strategy. This is why the ablation scores every arm on metrics no arm minimises,
and why the generalization experiment scores plans under weights they were not
chosen with — under mismatched weights the planner minimises a different function
than it is scored on, and the guarantee evaporates.

## Uncertainty

`sample_worlds` draws perturbed inputs; `PerturbationModel` controls the spreads.
Sensitivity and exposure are triangular around their nominal value, lifetime and
edge reliability multiplicative uniform. Quantum-factor and weight uncertainty are
**opt-in**, because switching them on changes what the reported interval means —
though they are the parameters the model is least sure about, and the shipped
experiments enable both.

Quantum-factor uncertainty is stored as a per-asset *multiplier*, not an absolute
override, and resolved against whatever algorithm the asset holds in the plan
being scored. An absolute override sampled from the pre-migration algorithm would
follow the asset through the migration and silently cancel it out.

### Common random numbers

Every strategy in a comparison is scored against the *same* sampled worlds. This
is the difference that matters most from 0.3, which used seeds 11, 12 and 13 for
the baseline, the heuristic and the optimizer, and printed three percentile bands
computed on different random draws side by side. Those bands were not comparable,
and their overlap said nothing about whether the strategies differed.

With shared worlds the per-world difference contains no sampling noise, and the
paired difference has far lower variance than either arm
(`tests/test_uncertainty.py::test_paired_deltas_have_lower_variance_than_either_arm`).

### Reporting

`paired_comparison` reports wins, losses, ties, a bootstrap CI on the mean
difference, and the mean difference **conditional on the plans differing**. All
four are needed. In this suite roughly two thirds of instances tie, so an
unconditional mean understates the effect where it exists, while a raw win rate
overstates how often the method applies.

Percentile bootstrap is used rather than a t-interval because paired differences
are heavily zero-inflated and far from normal.

## Experiments

| experiment | question | falsifiable? |
|---|---|---|
| `benchmark` | full report for one case | no — descriptive |
| `ablation` | is the benefit path-awareness or search power? | yes, on held-out metrics |
| `generalization` | do plans survive misspecified weights? | yes, no guarantee anywhere |
| `sensitivity` | is the recommendation stable across the simplex? | yes |

Every report embeds the weights, seeds, perturbation model, generator config and
algorithm registry that produced it, so a number can be traced to its assumptions.

## Scope

This is a research instrument. The quantum factors and objective weights are
uncalibrated scenario parameters encoding an ordering hypothesis, not measured
quantities. The evaluation instances are synthetic. A lower objective means a plan
scores better under this model; it does not mean a system is safer. Calibration
against real inventories and incident data is the missing step, and internal
consistency is not a substitute for it.

## Relation to crypto-agility guidance

NIST's crypto-agility work frames agility as replacing or adapting cryptographic
algorithms across protocols, applications, software, hardware, firmware and
infrastructure while preserving security and operations. Q-SHIELD takes the same
framing — migration as a constrained systems problem over a dependency graph
rather than an algorithm-replacement checklist — and asks whether reasoning over
that graph measurably beats ranking assets in isolation. It implements no standard
and claims no conformance.
