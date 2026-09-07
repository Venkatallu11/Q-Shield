"""Threat classes, and the time horizon each one is actually exposed over.

Every version of Q-SHIELD through 0.4 scored every asset against a single
``data_lifetime_years / 20`` longevity term. That term encodes **harvest now,
decrypt later**: an adversary records ciphertext today and decrypts it once a
cryptographically relevant quantum computer (CRQC) exists, so the risk grows
with how long the data must stay secret.

That reasoning does not transfer to signatures. Recording a signature gains an
adversary nothing — there is no ciphertext to open later, and a signature already
verified before a CRQC existed stays verified. Forgery only becomes possible from
the moment the CRQC exists, forward. Scoring a 90-day TLS leaf certificate as if
it carried the 15-year retention horizon of the data behind it overstates its
risk by more than an order of magnitude, and — worse — flattens the distinction
between that leaf and the 20-year root certificate authority above it, which is
the asset that actually matters.

Three classes, three horizons:

``CONFIDENTIALITY``
    Harvest now, decrypt later. Horizon = how long the plaintext must stay
    secret, regardless of how long the key is in service. Rotating the key does
    not help: the ciphertext is already recorded.

``AUTHENTICATION``
    Live forgery. Horizon = how long the credential remains *in service* and
    unrotated once a CRQC exists. A five-minute OIDC token is near-immune; a
    twenty-year offline root CA is maximally exposed. Rotation is the mitigation,
    which is why it is the one class where rotation capability genuinely reduces
    risk rather than merely indicating operational strain.

``NON_REPUDIATION``
    Long-term verifiability. Horizon = how long a signed *artifact* must remain
    checkable — archived code signatures, notarised documents, timestamped
    records. Retroactive in a different sense from confidentiality: an adversary
    with a CRQC can mint an artifact that appears to have been signed today.
    Rotating the signing key does not repair artifacts already in circulation.

The practical consequence, which
:mod:`qshield.experiments.threat_class` measures rather than asserts: correcting
the classification changes which assets a fixed budget should be spent on.
"""

from __future__ import annotations

from enum import Enum

from .algorithms import Primitive, primitive

# Horizon at which an exposure term saturates. Shared across classes so the three
# are on one scale and their weights stay comparable.
HORIZON_YEARS = 20.0

# Floor on the longevity factor. Nothing is entirely unexposed: even an ephemeral
# credential is forgeable during its lifetime.
MIN_LONGEVITY = 0.1


class ThreatClass(str, Enum):
    CONFIDENTIALITY = "confidentiality"
    AUTHENTICATION = "authentication"
    NON_REPUDIATION = "non-repudiation"


def default_threat_class(algorithm: str) -> ThreatClass:
    """The class an asset falls into if it does not declare one.

    Signature primitives default to ``AUTHENTICATION`` — the common case is a
    credential in live service. Assets whose signatures must outlive their keys
    (code signing, archival, notarisation) must say so explicitly, because
    nothing in the algorithm name distinguishes them.

    Key-establishment and symmetric primitives default to ``CONFIDENTIALITY``.
    """
    kind = primitive(algorithm)
    if kind is Primitive.SIGNATURE:
        return ThreatClass.AUTHENTICATION
    return ThreatClass.CONFIDENTIALITY


def longevity(
    threat_class: ThreatClass,
    *,
    data_lifetime_years: float,
    credential_validity_years: float | None,
    verification_horizon_years: float | None,
) -> float:
    """Exposure factor in ``[MIN_LONGEVITY, 1]`` for the relevant horizon.

    Falls back to ``data_lifetime_years`` when a class-specific horizon is not
    supplied, which keeps pre-0.5 case files loadable — but the fallback
    reproduces exactly the conflation this module exists to correct, so callers
    that care should set the horizon their assets actually face.
    """
    if threat_class is ThreatClass.AUTHENTICATION:
        horizon = credential_validity_years
    elif threat_class is ThreatClass.NON_REPUDIATION:
        horizon = verification_horizon_years
    else:
        horizon = data_lifetime_years
    if horizon is None:
        horizon = data_lifetime_years
    return min(1.0, max(MIN_LONGEVITY, horizon / HORIZON_YEARS))


def rotation_is_mitigating(threat_class: ThreatClass) -> bool:
    """Whether being able to rotate the key actually reduces this class of risk.

    Only for ``AUTHENTICATION``. Rotating a key does not un-harvest recorded
    ciphertext, and it does not repair signed artifacts already in circulation.
    Q-SHIELD's rotation term is a measure of operational pressure and is applied
    to every class; this predicate exists so that anything claiming rotation as a
    *mitigation* has to be explicit about where that claim holds.
    """
    return threat_class is ThreatClass.AUTHENTICATION
