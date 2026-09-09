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

## 12. The central recommendation survives total parameter ignorance (0.6)

Ingesting a real certificate estate produces a case in which roughly **55% of the
model's inputs are placeholders**. No certificate records how sensitive the data
behind a key is, how exposed the host is, or what migrating it would cost.

`qshield robustness` resamples every one of those fields across its full
admissible range — sensitivity and exposure over most of the unit interval, data
lifetime from 1 to 25 years, migration cost from 0.4x to 2.5x — and replans each
draw. Observed and derived fields (algorithm, credential validity, the trust
hierarchy, issued-certificate counts) are held fixed, because those are facts.

1000 draws, on a constructed three-tier CA (`cases/pki_case.json`) and on a real
certificate estate parsed from PEM files (`results/robustness_ingested.json`):

| tier | affordable in | chosen when affordable |
|---|---|---|
| root CA | 28.7% / 34.8% of draws | **91.3% / 83.3%** |
| intermediate CA | 76.1% / 100% | 21.8% / 34.3% |
| TLS leaf certificates | **100%** of draws | **0.8% – 3.5%** |

Leaf certificates are affordable in every single draw and are chosen in under 4%
of them. The root is chosen almost every time it can be paid for.

This is the independent confirmation of section 11. The hierarchy result was
derived from a model with chosen parameters; here it survives replacing every
choosable parameter with a uniform draw over its whole plausible range, on a real
estate as well as a constructed one. **The recommendation to migrate the trust
anchor rather than the leaves does not depend on the uncalibrated parameters at
all**, which is what makes an uncalibrated model actionable.

Note the secondary finding, identical on both estates: the root came back
**blocked by budget** — chosen ~90% of the times it was affordable, affordable
under a third of the time. The binding constraint was money, not analysis.

## 13. Calibration: two of the invented constants were wrong (0.7)

The parameters had been called "uncalibrated hypotheses" since 0.3. That was
honest but lazy — it lumped three different kinds of number together, and by
treating them all as ungroundable never checked whether the groundable ones were
right. Two were not. Full detail and sources in
**[CALIBRATION.md](CALIBRATION.md)**.

- **Elliptic curve is a much easier quantum target than RSA at equal classical
  security** — 2.6x fewer logical qubits, ~148x fewer Toffoli gates
  (Roetteler et al., ASIACRYPT 2017). Every version through 0.6 scored
  `RSA-2048`, `ECDSA`, `X25519` and `ED25519` at exactly 1.00, asserting they fall
  together. An estate that used this model to sequence RSA before ECDSA sequenced
  them backwards.
- **Symmetric cryptography is not meaningfully threatened.** NIST makes AES-128
  the benchmark for post-quantum security; AES-256 key search costs about
  2^298/MAXDEPTH gates. The registry scored AES-128 at 0.15 and AES-256 at 0.08 —
  *above* the 0.05 it gave ML-KEM. The model rated symmetric cryptography riskier
  than the post-quantum algorithms it recommends migrating to.
- **Two invented factors became one probability.** Quantum factor x longevity is
  replaced by P(the capability to break this primitive exists before this asset's
  exposure horizon ends), anchored on the Global Risk Institute / evolutionQ
  expert elicitation (26 experts: CRQC within 10 years 28-49%, within 15 years
  51-70%) and shifted per primitive by published resource estimates.

**Migration stops always being worthwhile.** Below a crossover horizon the chance
a CRQC arrives at all is smaller than the residual risk of a young post-quantum
primitive: `ECDSA -> ML-DSA` pays off only above **3.7 years**, `RSA-2048 ->
ML-KEM` above 5.3, `AES-256` never. A 90-day certificate is 0.25 years, and on
both the constructed hierarchy and the real ingested estate **half the migratable
certificates sit below the crossover**. This gives up the monotonicity guarantee
on the calibrated path, deliberately and with tests pinning it.

**Did it change the answer? Mostly no, and that is the honest headline.** Plan
agreement with the uncalibrated model is 93.3% on PKI instances and 89.7% on
generic ones, and on both real-shaped estates the recommendation is *identical*.
The trust-hierarchy model already excluded leaf certificates because their risk is
pinned to an unmigrated issuer; the calibrated model excludes them again because
their horizons are too short. Two unrelated lines of argument converge on
"migrate the anchor, leave the leaves", and neither needed the other.

Across all 36 combinations of the parameters calibration could *not* ground
(forecast scenario x hardware doubling time x PQC residual risk), plan agreement
averages **95.4%**, worst case **85.0%**. The recommendation is robust; the
crossover horizon is not, ranging from 0.05 to 5.9 years over the same settings.
Quote the plan, not the number.

## 14. The objective is supermodular, so greedy gets no guarantee (0.8)

A reviewer proposed testing the risk-reduction function for **submodularity**,
hoping it would hold: a monotone submodular objective hands greedy maximisation
the classic `1 - 1/e` bound, which would upgrade "greedy matched the optimum on
80.3% of instances" from an observation into a theorem.

It fails, and it fails in the direction that makes the bound unavailable.
Measured over 200 instances per suite with 40 nested set pairs each
(`results/curvature.json`):

| suite | submodularity violated | supermodularity violated |
|---|---|---|
| PKI with delegation | 38.8% of pairs | **0%** |
| generic, no delegation | 67.7% of pairs | **0%** |

The objective is **supermodular** — increasing returns — on every pair sampled,
in both suites. Migration actions are *complements*, not substitutes.

**The cause is more general than trust hierarchies.** Path risk composes as a
noisy-OR, `risk = 1 - prod(1 - p_v)`, so the benefit of migrating one node is
proportional to `prod(1 - p_v)` over the *other* nodes on the path. That factor
grows as those nodes are migrated, and the effect compounds with path length —
about 2.4x at length two, 35x at length five. Delegation is simply the limiting
case: a leaf's benefit is exactly **zero** until its anchor moves, and positive
afterwards.

So no approximation guarantee of that family applies to Q-SHIELD's objective, and
the 80.3% greedy-optimal rate stays an empirical observation. The negative result
is worth more than the theorem would have been: it says *why* greedy has no
bound here, and it is a property of the noisy-OR composition rather than of this
particular model.

**The follow-up hypothesis failed too.** If complementarity explained where greedy
loses, it would be a useful diagnostic. It does not: Spearman correlation between
an instance's violation rate and greedy's optimality gap is **0.019** (PKI) and
**0.143** (generic), and the optimality rates split by median complementarity have
overlapping intervals (0.859 [0.788, 0.919] against 0.750 [0.660, 0.830]). The
curvature finding is theoretically real and operationally inert as a predictor.

## 15. Hybrid deployments rescue the short-lived credentials (0.8)

Section 13 left an uncomfortable conclusion: under calibration, migrating a
90-day certificate to pure post-quantum *raises* modelled risk, because below a
3.7-year crossover the chance a CRQC arrives is smaller than the residual risk of
a young primitive. The model's advice was to leave most of a certificate estate
alone.

That was an artifact of offering one destination. A hybrid is broken only if
**both** halves are, so it is covered against the CRQC *and* against cryptanalysis
of the young primitive. Composing the two probabilities moves the crossover:

| migration | to pure PQC | to hybrid |
|---|---|---|
| `ECDSA` | 3.7 years | **0.15 years** |
| `ED25519` | 3.7 years | **0.15 years** |
| `X25519` | 3.7 years | **0.15 years** |
| `RSA-2048` | 5.3 years | **1.8 years** |

Eight weeks is shorter than a TLS leaf certificate's life, so the advice inverts:
migrate them, to a hybrid.

**Whether that changes the plan depends entirely on the estate**
(`results/hybrid_ladder.json`, 300 instances per suite):

| suite | pure policy | ladder policy |
|---|---|---|
| flat, short-lived credentials | **0 assets migrated**, 0 risk reduction | 2.58 assets, 0.031 reduction |
| | 0 of 1643 short-lived credentials | **775 of 1643** |
| PKI with delegation | 3.546 reduction | 3.560 (+0.4%) |
| | 0 of 1500 short-lived | 0 of 1500 |

On a flat estate of workload identities and service certificates, the
pure-post-quantum policy is **completely paralysed** — across 300 instances it
migrates nothing at all — and hybrid unblocks it. On a certificate hierarchy it
changes almost nothing, because trust inheritance had already excluded those
leaves for an entirely unrelated reason.

Two of this project's findings interact, and one subsumes the other: **where a
trust hierarchy exists, the delegation effect dominates the crossover effect.**
The crossover matters where there is no anchor above the credential.

Given the choice, the ladder took the hybrid rung **every time** (866 of 866, and
782 of 782). The advantage holds while the composition is safer than the primitive
it protects; a `composition_risk` above the PQC residual removes it, and that
parameter is Tier C.

## 16. Independent hops was a defect, not a caveat (0.9)

`METHODOLOGY.md` carried "path hops are treated as independent" as a limitation
from 0.4 onward. 0.5's delegation work turned it into a defect. Two sibling
certificates issued by one authority sit on an attack path;
`effective_node_risk` correctly raises both to the issuer's risk — forge the
root and you forge both — and then `path_risk` multiplies their probabilities as
though they were independent events. **They are the same event.** On the worked
case in `tests/test_correlation.py` that reports 31.8 where the joint
distribution gives 19.7, and the overstatement grows with each additional sibling
on the path.

0.9 stops composing marginals and composes the generative model instead. Each
asset carries its own compromise probability; each cause — an issuing authority,
a shared HSM, an identity provider, a cloud account, a library — compromises its
dependents with a propagation strength; path risk is the expectation over the
joint state of the causes. With no causes present it collapses to the previous
formula exactly, which is asserted by a test rather than assumed. A new
`EdgeKind.SHARED` expresses substrate as distinct from delegated trust.

**Fixing it properly moves two things in opposite directions**, so the net would
hide what happened (`results/correlation_impact.json`, 200 instances per suite):

| effect | what changed | pki delegation | shared substrate |
|---|---|---|---|
| **marginal** | max-dominance replaced by noisy-OR over own risk and causes | **−6.20** [−6.29, −6.11] | **−10.51** [−10.83, −10.20] |
| **correlation** | independent hops replaced by conditioning | 0.00 | **+18.25** [17.37, 19.15] |
| net on objective | | −6.20 | −5.03 |

The marginal effect **raises** risk: taking the max of own and inherited risk
discards an asset's own contribution whenever its issuer dominates, and about
70–80% of these assets' risk turns out to be inherited. The correlation effect
**lowers** path risk: on shared substrate the independent product overstates by
**36.4%** (50.1 against 31.9).

**And it matters exactly where two dependents of one cause share a path.** The
PKI suite's correlation effect is precisely 0.00 — not small, zero — because
those paths run `leaf -> service -> service` and contain only one certificate
each, so there are no correlated siblings to double-count. That condition is
stated as a test. Where services share an HSM and an identity provider, the same
correction is worth a third of the path term.

Plans disagree on **79%** of PKI instances and **63.5%** of shared-substrate ones,
so this is not a rescaling — it changes what to migrate.

## 17. A cause over the causes — and the first ground truth (0.10)

0.9 conditioned path risk on shared causes and left one gap: its cause model was
**flat**. Every cause was an independent source, and a cause standing behind
another — one supplier behind two hardware modules, one cloud region behind two
identity providers — was folded in by accumulating influence along the chain.

**Checking that against simulation is the first ground truth this project has
had.** Every earlier result checks internal consistency: the objective against
its own definition, a planner against exhaustive enumeration. But the cause model
states a *generative* story — assets self-compromise, cause edges fire, a node
falls if it self-compromises or a live parent's edge fires — and that story can be
simulated directly. Analytic and simulation are then two independent computations
of one number.

0.9 failed it. **Exact whenever propagation was certain; biased upward by up to a
full risk point, at nine standard errors, whenever propagation was partial.** Two
errors in the same place that cancel at strength 1.0: the transitive term gave an
upstream cause a second, independent shot at a node through a channel its child
had already carried, and the state weight treated a cause and its own parent as
independent, leaving weight on incompatible states. No amount of internal
consistency would have found either.

0.10 replaces it with a noisy-OR Bayesian network on the cause DAG, enumerating
cause states in topological order using **direct** parents only, so each channel
counts exactly once. Cross-signed authorities are collapsed into a single cause —
if two mutually certify, compromising either compromises both — and that decision
is itself checked against a fixpoint simulation rather than asserted. The
simulator shares no code with the analytic path
(`tests/test_correlation_montecarlo.py`).

Writing that test immediately caught a second bug: when a node is itself one of
the causes being conditioned on, its state is *decided* by the cause state, and
recomputing its own risk counted the self-compromise twice. It reported 23.9
where simulation said 14.5.

**What the correction is worth** (`results/correlation_impact.json`, 200
instances per suite):

| suite | independent hops | conditioned | overstated by |
|---|---|---|---|
| PKI delegation | 28.47 | 28.47 | 0.0% |
| shared substrate | 50.15 | 31.90 | **36.4%** |
| supply chain (causes over causes) | 29.32 | 25.27 | **13.8%** |

The supply-chain row is the one 0.9 could not have produced correctly.

**And the honest negative that follows.** Take two identical estates — same
inventory, same budget, same candidate set — where one model records that the
modules share a supplier and the other does not:

| | |
|---|---|
| path risk understated by omitting it | **3.02** (p05 1.07, p95 5.70) |
| plan agreement | **99%** |
| risk reduction forgone by the blind plan | **~0** |

Not recording a shared supplier makes you understate your risk by about three
points **and does not change what you should do.** The reason is structural: a
supplier is a cause you cannot buy — you cannot migrate someone else's firmware —
so knowing about it changes the number, not the menu of actions. It matters for
assessment and not for prioritisation, which is worth knowing before anyone
spends effort mapping their supply chain to feed this model.

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
- **Section 12 shows robustness, not correctness.** That a recommendation
  survives every setting of the *guessed* parameters says nothing about whether
  the *model* is right. The quantum factors, the objective weights and the
  max-dominance trust semantics are held fixed throughout and remain
  uncalibrated; a robust answer from a wrong model is still wrong.
- **Calibration is substitution of published estimates, not validation.** No
  CRQC exists, nothing has been compromised via Shor's algorithm, and so there
  are zero outcome events to fit against. Section 13 replaces invented constants
  with other people's published numbers. That is a real improvement and it is not
  the same as a model checked against reality.
- **The expert survey is an elicitation.** It records what 26 people believed in
  2025, and its own authors publish it as a range because they disagree.
- **Supermodularity is measured on sampled pairs, not proved.** Section 14
  reports that no sampled pair violated supermodularity; that is strong evidence
  and not a proof over all subsets. The noisy-OR argument explains the mechanism
  but is not a formal theorem covering the full objective, which also carries
  node and rotation terms.
- **The hybrid composition assumes independent breaks.** A CRQC solving
  elliptic-curve discrete log gives no purchase on a lattice problem, so the
  product is defensible for the *mathematics*; correlated **implementation**
  failure is real and is priced as `composition_risk`, which is itself an
  assumption.
- **The generative model is validated; the model itself is not.** Section 17
  shows the analytic code computes what the cause model says. Whether that model
  describes real infrastructure is a separate question no simulation can settle.
- **Cause edges fire independently.** A supplier reaching two modules is now
  modelled, but the two firings are independent draws. A defect that
  deterministically affects every unit of one firmware build is not the same
  thing, and is not represented.
- **Correlated mode is opt-in.** Sections 1-15 were measured under the
  independent-hops model and are unchanged; they are internally consistent and,
  where a path carries two dependents of one cause, they overstate the path term.
- **The ingested estate is a test fixture.** It is a real, correctly-signed
  OpenSSL hierarchy, not a production estate, and it is small. The agreement
  between it and the constructed case is encouraging, not conclusive.
