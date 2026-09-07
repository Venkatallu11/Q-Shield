# Calibration

**Q-SHIELD is not calibrated against outcomes, and cannot be.**

Calibration in the strict sense means fitting parameters to observed results. No
cryptographically relevant quantum computer exists. Nothing has ever been
compromised via Shor's algorithm. There are exactly zero outcome events to fit
against, and no amount of engineering produces data that does not exist. Any tool
advertising a *calibrated* quantum risk score is claiming something nobody has.

What 0.7 does instead is narrower and still worth doing: it replaces invented
constants with **published quantitative estimates**, and separates the parameters
that can be grounded that way from the ones that cannot. Until now the docs
called every parameter "an uncalibrated hypothesis", which was honest but lazy —
it lumped three different kinds of number together, and by treating them all as
ungroundable it never checked whether the groundable ones were even right.

They were not. Two of them were wrong, and one had the sign backwards.

---

## The three tiers

| tier | what | grounded in | status |
|---|---|---|---|
| **A** | relative quantum work per primitive; CRQC arrival probability | resource-estimate papers; a seven-year expert elicitation | published numbers, cited below |
| **B** | hardware progress rate converting a resource ratio into a time offset | — | **a stated assumption**, swept |
| **C** | objective weights, sensitivity, exposure, criticality, migration cost, PQC residual risk | — | **unobservable**, handled by `qshield robustness` |

Tier C is not a gap calibration can close. No literature will ever tell you how
sensitive *your* data is or what *your* migration costs. That is what the
robustness pass in [USING.md](USING.md) is for.

## What the calibration changed

### 1. Elliptic curve is a much easier quantum target than RSA

At equal classical security (~128 bits), Roetteler et al. give:

| primitive | logical qubits | Toffoli gates |
|---|---|---|
| ECC-256 | **2,330** | **1.26 × 10¹¹** |
| RSA-3072 | 6,146 | 1.86 × 10¹³ |

ECC needs **2.6× fewer logical qubits and ~148× fewer Toffoli gates**. Proos and
Zalka reached the same conclusion earlier: breaking ECC is easier than breaking
RSA at comparable classical strength.

Every Q-SHIELD version through 0.6 scored `RSA-2048`, `ECDSA`, `X25519` and
`ED25519` at exactly **1.00** — asserting they all fall at the same instant. They
do not. **An estate that used this model to sequence RSA before ECDSA sequenced
them in the wrong order.** Under calibration ECC leads RSA-2048 by ~1.6 years,
and at a 5-year horizon it scores roughly 5× higher.

### 2. Symmetric cryptography is not meaningfully threatened

Grover gives a quadratic speedup, and a quadratic speedup on a 128-bit key is not
an attack. NIST makes **AES-128 the benchmark** against which post-quantum
candidates are measured; key search on AES-256 costs on the order of
**2²⁹⁸/MAXDEPTH** quantum gates, and Grover parallelises poorly, so the depth
restriction bites hard.

The shipped registry scored `AES-128` at **0.15** and `AES-256` at **0.08** —
against **0.05** for `ML-KEM`. The model was rating symmetric cryptography as
carrying *more* quantum risk than the post-quantum algorithms it recommends
migrating to. That is backwards, and it inflated the baseline risk of every
data-at-rest asset in every case file. Calibration sets them to zero.

### 3. Two invented factors become one probability

The old node risk multiplied a quantum factor by a longevity term. Neither
denoted anything. The calibrated form computes

> **P(the capability to break this primitive exists before this asset's exposure
> horizon ends)**

anchored on published expert elicitation and shifted per primitive by published
resource estimates. It is a probability about the world, so it can be argued
with — which the old product could not be.

The arrival curve comes from the Global Risk Institute / evolutionQ **Quantum
Threat Timeline Report 2025** (Mosca & Piani, 26 experts), the longest-running
elicitation on the question: a CRQC within 10 years at **28–49%**, within 15 years
at **51–70%**. Rather than collapse those ranges to a midpoint and present it as
precise, both endpoints ship as separate scenarios (`conservative`, `central`,
`aggressive`) and the experts' disagreement is propagated into the results.

## The consequence nobody asked for: migration is not always net-positive

Because the calibrated term is a probability over a horizon, a short-lived
credential can be *worse off* after migrating. Below a crossover horizon, the
chance a CRQC arrives at all is smaller than the residual risk of a young
post-quantum primitive:

| migration | pays off above |
|---|---|
| `ECDSA → ML-DSA` | 3.7 years |
| `X25519 → ML-KEM` | 3.7 years |
| `RSA-2048 → ML-KEM` | 5.3 years |
| `AES-256 → ML-KEM` | never |

A 90-day TLS certificate is 0.25 years. **On both the constructed CA hierarchy
and the real ingested estate, half the migratable certificates sit below the
crossover.** The model now says: leave them alone, and spend the budget on the
anchor.

This breaks a guarantee. `qshield.model` documents monotonicity — migrating to a
less susceptible algorithm never raises the score — and the calibrated path gives
that up deliberately. It is a finding, not a defect, and it is pinned by tests in
`tests/test_monotonicity.py`. The planners are unaffected: exhaustive enumeration
includes the null plan and greedy stops when no move helps, so neither ever
relied on monotonicity. Only the theorem did.

**Its location is not trustworthy.** The *existence* of a crossover follows from
the calibrated form. Where it sits depends on `pqc_residual_risk`, which is Tier C
and uncalibrated — sweeping it from 0.01 to 0.10 moves the ECDSA crossover from
0.05 to 5.9 years. Do not plan against "3.7 years". Plan against the plan.

## Does calibration change the answer?

`results/calibration_impact.json`, 300 instances per suite:

| suite | plan agreement with the uncalibrated model | assets gaining priority | assets losing it |
|---|---|---|---|
| PKI hierarchies | **93.3%** | elliptic-curve 14 | elliptic-curve 8, RSA 3 |
| generic estates | **89.7%** | elliptic-curve 30, RSA 4 | RSA 21, elliptic-curve 20 |

The direction is the mechanism showing through: elliptic-curve assets gain
priority and RSA loses it, which is exactly what correcting error 1 predicts.

And on the two real-shaped estates — the ingested certificate inventory and
`cases/pki_case.json` — the recommendation is **identical**.

That is the honest headline, and it is not the exciting one: **calibration
corrected two real errors in the registry and added a new quantitative
constraint, but mostly confirmed the recommendations already being made.**

The reason is worth noting. The trust-hierarchy model already excluded leaf
certificates, because their risk is pinned to an unmigrated issuer. The
calibrated model excludes them again, for an entirely unrelated reason: their
horizons are below the crossover. Two independent lines of argument converge on
the same instruction — *migrate the anchor, leave the leaves* — and neither
needed the other.

## How much of the calibrated answer is still assumed?

`results/calibration_sensitivity.json` sweeps all 36 combinations of forecast
scenario (3) × hardware doubling time (3) × PQC residual risk (4):

```
plan agreement with the default calibration:
  mean 95.4%   p05 86.5%   worst case 85.0%
```

So the *recommendation* is robust to everything the calibration could not ground.
The *crossover horizon* is not — it ranges from 0.05 to 5.9 years across the same
settings. Quote the plan; do not quote the number.

## Sources

- Roetteler, Naehrig, Svore & Lauter, *Quantum resource estimates for computing
  elliptic curve discrete logarithms*, ASIACRYPT 2017 — [arXiv:1706.06752](https://arxiv.org/abs/1706.06752)
- Gidney & Ekerå, *How to factor 2048 bit RSA integers in 8 hours using 20 million
  noisy qubits*, Quantum 5, 433 (2021) — [arXiv:1905.09749](https://arxiv.org/abs/1905.09749)
- Gidney, *How to factor 2048 bit RSA integers with less than a million noisy
  qubits* (2025) — [arXiv:2505.15917](https://arxiv.org/abs/2505.15917)
- Jaques, Naehrig, Roetteler & Virdia, *Implementing Grover oracles for quantum
  key search on AES and LowMC*, EUROCRYPT 2020 — [arXiv:1910.01700](https://arxiv.org/abs/1910.01700)
- NIST, *Post-Quantum Cryptography: security strength categories* — [csrc.nist.gov](https://csrc.nist.gov/projects/post-quantum-cryptography/faqs)
- Mosca & Piani, *Quantum Threat Timeline Report*, Global Risk Institute /
  evolutionQ, 2024 and 2025 editions — [globalriskinstitute.org](https://globalriskinstitute.org/publication/2024-quantum-threat-timeline-report/), [evolutionq.com](https://www.evolutionq.com/publications/quantum-threat-timeline-research-report-2025)

## What is still not calibrated

- **Outcomes.** Nothing here has been validated against a real compromise,
  because none has occurred. This is substituted estimates, not a validated model.
- **The hardware progress rate** (Tier B) that converts a resource ratio into a
  time offset.
- **Every Tier C parameter**: objective weights, per-asset sensitivity, exposure,
  business criticality, migration cost, and the PQC residual risk that sets the
  crossover.
- **The trust semantics.** That a forged root certificate forges everything
  beneath it is a cryptographic fact; modelling it as complete max-dominance is a
  choice.
- **The expert survey is an elicitation, not a measurement.** It records what 26
  people believed in 2025. Its own authors present it as a range because they
  disagree with each other.
- **Quantum risk only.** Phishing, credential theft, key mismanagement and
  misconfiguration remain unmodelled and remain likelier.
