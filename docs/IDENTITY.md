# Identity infrastructure

Identity is where post-quantum migration is hardest to reason about and easiest
to get wrong. This document covers what Q-SHIELD 0.5 added to model it, what the
experiments measured, and — importantly — what quantum risk is *not* the dominant
threat to.

All numbers are reproducible via `scripts/run_all.sh`; reports are in `results/`.

---

## Two modelling errors that specifically break identity

### 1. Harvest-now-decrypt-later does not apply to signatures

Every version through 0.4 scored every asset's longevity as
`data_lifetime_years / 20`. That term encodes harvest-now-decrypt-later: an
adversary records ciphertext today and decrypts it once a CRQC exists, so risk
grows with how long the plaintext must stay secret.

Signatures do not work that way. Recording a signature gains an adversary
nothing; a signature verified before a CRQC existed stays verified. Forgery
becomes possible only from the moment the CRQC exists, *forward*. The horizon
that matters is how long the credential stays **in service**, not how long the
data behind it is retained.

The consequence in a certificate hierarchy is severe. A 90-day TLS leaf
certificate and the 20-year root CA above it sit behind the same 15-year data
retention horizon, so the old term scored them **identically**. Under the
corrected horizons they differ by a factor of ten.

Q-SHIELD 0.5 splits three classes ([`qshield/threat.py`](../src/qshield/threat.py)):

| class | horizon | rotation helps? | example |
|---|---|---|---|
| `CONFIDENTIALITY` | how long the plaintext must stay secret | no — the ciphertext is already recorded | TLS key exchange, encrypted backups |
| `AUTHENTICATION` | how long the credential stays in service | **yes** — forgery is a live attack | CA keys, IdP token signing, passkeys |
| `NON_REPUDIATION` | how long the artifact must remain verifiable | no — artifacts in circulation are not repaired | code signing, notarised documents |

Signature primitives default to `AUTHENTICATION`; everything else to
`CONFIDENTIALITY`. A signing key whose artifacts outlive it must say so
explicitly, because nothing in the algorithm name distinguishes code signing from
a login credential.

**Measured** (`results/threat_class.json`, 300 PKI instances). The corrected
scoring does not merely re-rank — it inverts the recommendation:

| selected assets | pre-0.5 scoring | threat-class scoring |
|---|---|---|
| roots | 33 | **175** |
| intermediates | 201 | 37 |
| leaves | **1153** | 8 |

Plans agree on **2.3%** of instances. Real risk reduction achieved: **1.205**
under the old scoring versus **4.290** under the corrected one, a shortfall of
3.085 (95% CI [2.665, 3.499]).

### 2. Trust is dominance, not traversal

Q-SHIELD had one edge type through 0.4. In a dependency graph an edge is
something an adversary must *traverse*: it costs a step, and every node on the way
is another thing that has to fail.

A certificate authority is not that. If a root CA's key is forgeable, every
certificate beneath it is forgeable — immediately, simultaneously, with nothing in
between to defeat. 0.5 adds `EdgeKind.DELEGATION` and computes

```
effective(v) = max(own(v), max over issuers u of effective(u) . strength(u, v))
```

by fixpoint iteration, because cross-signed hierarchies contain cycles. Being a
max of monotone terms, it preserves the objective's monotonicity.

**0.9 supersedes this in correlated mode, because the max was only half right.**
Two defects: taking the max of an asset's own risk and its inherited risk
discards the asset's own contribution whenever the issuer dominates, and — worse —
raising two siblings to the issuer's risk and then multiplying them through the
path term counts one compromise twice. `qshield.correlation` replaces both with a
generative model: a noisy-OR marginal over own risk and each cause, and a path
term conditioned on the joint state of the causes. `EdgeKind.SHARED` extends the
same treatment to substrate — one HSM or identity provider behind several
assets — which is a common cause without being a trust relation. Measured
effects are in [FINDINGS.md](FINDINGS.md#16-independent-hops-was-a-defect-not-a-caveat-09);
the correction is worth 36.4% of the path term on shared-substrate estates and
exactly nothing on the certificate hierarchies below, because those paths carry
only one certificate each.

## The crypto-agility trap

The two corrections combine into one operational conclusion:

> **Migrating leaf certificates while their issuer still signs with a classical
> algorithm reduces risk by exactly zero.**

The spend is real. The effective risk of every leaf stays pinned to the
unmigrated root. And a pre-0.5 planner, seeing cheap high-exposure leaves and an
expensive low-exposure root, buys the leaves — which is what makes this a trap
rather than merely an omission.

**Measured** (`results/hierarchy.json`, 300 PKI instances). Every plan is judged
in the delegation-aware model:

| planner | real improvement | improvement it *believed* it got | illusory | budget wasted | instances where the **entire** spend bought nothing |
|---|---|---|---|---|---|
| `flat_priority` | 0.003 | 2.021 | **99.9%** | 99.4% | **297 / 300** |
| `blind_omits_trust` | 1.138 | 2.202 | 89.2% | 88.8% | 265 / 300 |
| `blind_trust_as_dependency` | 1.138 | 2.202 | 89.2% | 88.8% | 265 / 300 |
| `hierarchy_aware` | **4.290** | 0.571 | 0.3% | 0.0% | 0 / 300 |

Trust-blind and trust-aware planning agree on **0 of 300** instances.

Note the fourth column. The blind planners report *more* improvement than the
aware one — 2.2 against 0.57 — while delivering a quarter of it. Modelling
delegation makes the reported number smaller and true.

This is deliberately **not** presented as a win rate. The hierarchy-aware planner
minimises the metric it is scored on and every rival plan is feasible for it, so
it cannot lose; that structural guarantee is precisely what invalidated 0.3's
headline result. What is reported is the *size of the modelling error*, which was
free to come out zero.

### When the trap does not bite

The magnitude above depends on the generator pricing a root at 7.5x a leaf. That
is realistic but it is an assumption, so it was swept
(`results/hierarchy_root_cost.json`, 200 instances per point):

| root cost | vs leaf | illusory fraction | blind improvement | aware improvement |
|---|---|---|---|---|
| 0.5 | 1.25x | 0.013 | 8.283 | 9.773 |
| 1.0 | 2.5x | 0.040 | 7.722 | 9.133 |
| 2.0 | 5.0x | 0.469 | 4.485 | 6.851 |
| 3.0 | 7.5x | 0.918 | 0.882 | 4.096 |
| 5.0 | 12.5x | 0.998 | 0.004 | 0.005 |
| 8.0 | 20x | 0.998 | 0.004 | 0.005 |

Three regimes, and the honest claim is the middle one:

- **Cheap anchors (under ~2.5x a leaf).** A blind planner buys the root anyway,
  for its own node risk. Delegation modelling changes little; the refinement is
  not worth its complexity here.
- **Anchors at 5–10x a leaf, still inside budget.** The trap bites hardest. This
  is the regime the claim is about.
- **Anchors beyond budget (12x+).** Neither planner achieves anything, because
  nothing can be bought. But the aware model *says so* (0.005), while the blind
  model reports 2.2 of improvement that is 99.8% fiction. Even where no action is
  possible, one model tells you the truth and the other sells you a plan.

### Worked example

`cases/pki_case.json` — three tiers, budget affording either the root or all
three leaves. A constructed illustration, not evidence:

| planner | selects | cost | real improvement |
|---|---|---|---|
| `flat_priority` | 3 leaves | 1.2 | **0.000** |
| pre-0.5 scoring | intermediate + 2 leaves | 1.8 | **0.000** |
| trust-blind exhaustive | intermediate + 2 leaves | 1.8 | **0.000** |
| Q-SHIELD 0.5 | `root-ca` | 2.0 | 2.604 |

Three different blind strategies spend real budget for exactly zero risk
reduction.

## Practical implications

1. **Migrate trust anchors first.** Not because they are the most exposed — they
   usually are not — but because nothing beneath them can improve until they do.
2. **Do not score signing keys by data retention.** Score them by how long the
   credential stays in service. A 20-year offline root and a 90-day leaf are not
   comparable assets.
3. **Distinguish signing keys whose artifacts outlive them.** Code signing and
   notarisation are `NON_REPUDIATION`: rotating the key does not repair
   signatures already in circulation.
4. **Treat a federated identity provider as a root CA.** Structurally it is one:
   compromising it compromises every relying party at once.
5. **Be suspicious of a large reported improvement.** In these experiments the
   blind planners consistently reported *higher* improvement than the correct
   model. A trust-aware model reports smaller, truer numbers.

## Personal identity

The same machinery applied to an individual (`results/personal_identity.json`,
300 instances), where account-recovery chains are the attack paths and a
federated identity provider is the trust anchor. Assets the person cannot change
— their bank's key exchange, a hospital's archive — are marked
`controllable=false`: their risk is still scored, because it is real, but no
budget can be spent on them.

| quantity | mean | p05 | p95 |
|---|---|---|---|
| share of modeled risk on assets the individual controls | **30.2%** | 18.2% | 44.4% |
| share of the objective removable within budget | **12.1%** | 1.5% | 25.8% |

Risk by threat class, before and after the individual does everything available
to them:

| class | before | residual |
|---|---|---|
| `CONFIDENTIALITY` | 26.16 | **20.86** |
| `AUTHENTICATION` | 10.39 | 9.14 |

Two conclusions follow, and neither is the flattering one:

**Most of an individual's modeled quantum risk is not theirs to fix.** Roughly 70%
sits on assets they do not control, and a realistic budget removes about 12% of
the objective. The gap between "risk present" and "risk addressable" is the whole
story, and any tool that reports only what its plan achieved will look far better
than the situation warrants.

**What remains is the part that cannot be deferred.** Residual risk is
concentrated 2.3:1 in confidentiality over authentication. That asymmetry matters
because the two classes age differently. Credential risk is remediable later —
forgery is a live attack, so a passkey replaced in 2030 is as good as one replaced
today. Harvested-data risk is not: traffic copied today is copied, and migrating
afterwards protects only future traffic. The individual's residual sits
overwhelmingly in the class where waiting is irreversible, and the lever that
matters is not credential hygiene but **which providers hold long-lived personal
data and when they deploy PQ key exchange** — genomic and health records being the
extreme case, since their sensitivity does not decay and, for genomic data,
implicates relatives who never made any choice at all.

## What Q-SHIELD does not model, and why it matters here

**Q-SHIELD models quantum risk and nothing else.** It does not model phishing,
credential stuffing, SIM-swap, session-token theft, malware, device theft, or
insider access. Between them those account for the overwhelming majority of real
identity compromise today, and none of them requires a quantum computer.

For personal identity in particular this is not a footnote. A person who reads
this document and migrates their passkey while reusing a password on their email
provider has made themselves no safer in any way that matters. The honest framing
is that the analysis here covers one narrow, slow-moving, irreversible threat —
harvested long-lived data — and that everything else about protecting an identity
lies outside what this model can see.

Further limits carry over from the base model: the scenario parameters are
uncalibrated, the instances are synthetic, and path risk assumes independent hops
where real infrastructure shares platforms and operators. See
[METHODOLOGY.md](METHODOLOGY.md#scope) and
[FINDINGS.md](FINDINGS.md#what-none-of-this-establishes).
