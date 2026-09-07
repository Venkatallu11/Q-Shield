# Using Q-SHIELD on a real estate

Everything before 0.6 ran on synthetic instances. This is the path from actual
infrastructure to a decision.

```bash
pip install -e ".[scan]"

# 1. Inventory: certificates on disk, live TLS endpoints, or both
qshield ingest --certs /etc/ssl/certs ./pki --budget 6 --output case.json
qshield ingest --tls api.example.com www.example.com:443 --budget 6 --output case.json

# 2. Ask whether the recommendation depends on what had to be guessed
qshield robustness --input case.json --draws 1000

# 3. Read the report
qshield report --input case.json --draws 1000 --output migration.md
```

---

## What ingest can and cannot know

The 0.5 model needed two things earlier versions could not express — how long a
signing credential stays in service, and which authority it inherits trust from —
and a certificate carries both exactly:

| model input | where it comes from | status |
|---|---|---|
| `algorithm` | subject public key info, with key size | **observed** |
| `credential_validity_years` | `notAfter − notBefore` | **derived** |
| delegation edges | issuer/subject, matched on AKI/SKI | **derived** |
| `dependency_count` | certificates this subject issued | **derived** |
| `threat_class` | `NON_REPUDIATION` from a codeSigning EKU, else `AUTHENTICATION` | **derived** |
| `sensitivity` | — | **assumed** |
| `exposure` | — | **assumed** |
| `data_lifetime_years` | — | **assumed** |
| `business_criticality` | — | **assumed** |
| `migration_cost` | — | **assumed** |
| `rotation_hours` | — | **assumed** |

**About 55% of the fields are assumptions.** No certificate records how sensitive
the data behind it is, how exposed the host is, or what migrating it would cost.

There are two bad responses to that and one good one. Presenting the score as a
measurement hides the guesswork. Refusing to produce a recommendation until
everything is calibrated means nobody ever gets one, because that data does not
exist. The third option is to stop asking what the parameters are and ask
**whether the answer depends on them**.

Every generated case therefore carries a `_provenance` block recording the origin
of every field, and every report prints the assumed share before its conclusions.

## The robustness pass

`qshield robustness` resamples every unobservable field across its full plausible
range — sensitivity and exposure over most of the unit interval, data lifetime
from 1 to 25 years, migration cost from 0.4× to 2.5× — replans each time, and
reports how often each asset is selected. Observed and derived fields are held
fixed: those are facts about the estate, and the question is what follows from
the facts alone.

Five verdicts, and the distinctions between them are the point:

| verdict | meaning | what to do |
|---|---|---|
| **act now** | chosen in ≥90% of draws | act; calibrating first would not change it |
| **likely** | chosen in 60–90% | reasonable to start on, worth confirming |
| **contested** | flips with the guesses | measure this asset's parameters |
| **blocked by budget** | chosen whenever affordable, rarely affordable | a funding decision, not a cryptographic one |
| **never affordable** | no draw could fit it | **not** a finding that it does not matter — it was never assessed |

The last two matter most in practice. "This would obviously be chosen if you
could pay for it" and "this was assessed and rejected" are different findings
that a single selection rate collapses together.

## The result that makes this usable

Run over 1000 redraws of every guessed parameter, on a constructed three-tier CA
(`cases/pki_case.json`) and on a real certificate estate ingested from PEM files
(`results/robustness_ingested.json`):

| tier | affordable in | chosen when affordable |
|---|---|---|
| root CA | 28.7% / 34.8% of draws | **91.3% / 83.3%** |
| intermediate CA | 76.1% / 100% | 21.8% / 34.3% |
| TLS leaf certificates | **100%** of draws | **0.8% – 3.5%** |

Leaf certificates are affordable in every single draw and are almost never worth
migrating. The root is chosen almost every time it can be paid for. **That
conclusion does not depend on any of the guessed parameters** — it survives their
entire admissible range, on both a constructed case and a real one.

So the central recommendation — *migrate the trust anchor; migrating leaves
beneath it buys nothing* — can be acted on today, without calibrating anything.
That is the whole argument for using an uncalibrated model, and it is why the
robustness pass is not an optional extra.

Note what it also says: on both estates the root came back **blocked by budget**.
The binding constraint was money, not analysis.

## Planner selection

`exhaustive` enumerates every budget-feasible subset, so it costs `O(2^n)`.
Measured (`scripts/scaling.py`, `results/planner_scaling.json`):

| migration candidates | exhaustive | greedy |
|---|---|---|
| 11 | 0.37 s | 0.014 s |
| 13 | 1.18 s | 0.014 s |
| 14 | 3.05 s | 0.025 s |
| 16 | 12.8 s | 0.028 s |
| 17 | 22.9 s | 0.030 s |

(Timings are wall-clock on one machine and will differ on yours; the plan counts
are deterministic.) Each additional candidate roughly doubles the exact cost. Above 16 candidates
the planner downgrades to greedy — which returns the exact optimum on 80.3% of
benchmark instances at a mean objective gap of 0.248 — and **says so in the
report**. A real certificate estate has hundreds of candidates, so this is the
normal path, not the exception.

## Certificate inventories have no attack paths

X.509 records trust relations, not which service reaches which. So an
ingested case has delegation edges and no dependency edges, and the path term of
the objective is inert: only node risk and trust inheritance are scored. The
report says so rather than printing "attack paths: 0" and letting you assume the
analysis ran.

To use the path term, add dependency edges and `--entrypoints`/`--targets` once
you know the service topology. Inventing them from certificate data would
manufacture a path term out of nothing.

## Honest limits

- **Assumed parameters are assumptions.** Any number depending on them is
  provisional. Use the robustness pass to find out which ones are.
- **Quantum risk only.** Not phishing, credential theft, key mismanagement,
  misconfiguration or weak parameters, which are likelier ways any of these
  assets is actually compromised. This is not a security assessment.
- **The scenario parameters are uncalibrated.** Quantum factors and objective
  weights encode an ordering hypothesis, not measurements.
- **An expired certificate that is still deployed is inventoried and flagged**,
  not dropped — that is the kind worth knowing about.
- **A missing issuer understates risk.** If a leaf's CA is not in the inventory,
  its trust inheritance is missing; the ingest warns, and the fix is to add the
  CA certificates.
- **Live TLS scanning reads what is presented, unverified.** Behind an
  intercepting proxy you will inventory the proxy's PKI rather than the origin's
  — which is itself worth knowing, but is not what you asked for.
