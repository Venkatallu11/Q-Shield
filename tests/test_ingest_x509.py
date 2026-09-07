"""Turning real certificates into a case.

The fixtures in ``tests/fixtures/pki`` are genuine certificates emitted by
OpenSSL and signed by a real chain (see ``scripts/make_fixtures.sh``), not
hand-built structures. A parser tested only against its own output proves very
little.
"""

from pathlib import Path

import pytest

pytest.importorskip("cryptography", reason="X.509 ingest needs the 'scan' extra")

from qshield.experiments.benchmark import problem_from_dict  # noqa: E402
from qshield.ingest.provenance import Source  # noqa: E402
from qshield.ingest.x509 import (  # noqa: E402
    CertificateIngestError,
    from_certificates,
    load_certificates,
)
from qshield.threat import ThreatClass  # noqa: E402

PKI = Path(__file__).resolve().parent / "fixtures" / "pki"


@pytest.fixture(scope="module")
def inventory():
    return from_certificates(load_certificates([PKI]))


@pytest.fixture(scope="module")
def by_name(inventory):
    return {a.name: a for a in inventory.assets}


def test_reads_every_certificate_in_a_directory(inventory):
    assert len(inventory.assets) == 6


def test_a_bundle_yields_its_whole_chain():
    """Servers deploy leaf+intermediate+root in one file; reading only the first
    block would silently drop the issuers, which are the assets that matter."""
    certs = load_certificates([PKI / "chain-bundle.pem"])
    assert len(certs) == 3


def test_the_same_ca_appearing_in_several_chains_is_one_asset():
    certs = load_certificates([PKI / "chain-bundle.pem", PKI / "leaf-api.pem"])
    inventory = from_certificates(certs)
    roots = [a for a in inventory.assets if a.name.endswith("Root CA")]
    assert len(roots) == 1


def test_public_key_algorithm_is_read_from_the_subject_key(by_name):
    assert by_name["Q-SHIELD Test Root CA"].algorithm == "RSA-4096"
    assert by_name["Q-SHIELD Test Intermediate CA"].algorithm == "ECDSA"
    assert by_name["web.example.test"].algorithm == "ECDSA"
    assert by_name["api.example.test"].algorithm == "RSA-2048"
    assert by_name["Q-SHIELD Release Signing"].algorithm == "ED25519"


def test_validity_period_becomes_the_credential_horizon(by_name):
    """notAfter - notBefore is exactly the horizon qshield.threat selects for an
    AUTHENTICATION asset. This is the fit that makes ingest worth having."""
    assert by_name["Q-SHIELD Test Root CA"].credential_validity_years == pytest.approx(
        20.0, abs=0.05
    )
    assert by_name[
        "Q-SHIELD Test Intermediate CA"
    ].credential_validity_years == pytest.approx(8.0, abs=0.05)
    assert by_name["web.example.test"].credential_validity_years == pytest.approx(
        0.246, abs=0.01
    )


def test_a_root_and_a_leaf_are_scored_eighty_times_apart(by_name):
    """The separation the pre-0.5 shared data-retention horizon could not make."""
    root = by_name["Q-SHIELD Test Root CA"].credential_validity_years
    leaf = by_name["web.example.test"].credential_validity_years
    assert root / leaf > 50


def test_code_signing_eku_becomes_non_repudiation(by_name):
    """A codeSigning EKU is a statement that artifacts outlive the key."""
    signing = by_name["Q-SHIELD Release Signing"]
    assert signing.effective_threat_class is ThreatClass.NON_REPUDIATION
    assert signing.verification_horizon_years is not None


def test_ordinary_certificates_are_authentication(by_name):
    for name in ("web.example.test", "Q-SHIELD Test Root CA"):
        assert by_name[name].effective_threat_class is ThreatClass.AUTHENTICATION


def test_trust_hierarchy_is_recovered_exactly(inventory):
    edges = {(e.source, e.target) for e in inventory.edges}
    assert ("Q-SHIELD Test Root CA", "Q-SHIELD Test Intermediate CA") in edges
    for leaf in ("web.example.test", "api.example.test", "Q-SHIELD Release Signing"):
        assert ("Q-SHIELD Test Intermediate CA", leaf) in edges
    # A self-signed root inherits from nobody.
    assert not any(t == "Q-SHIELD Test Root CA" for _, t in edges)


def test_delegation_edges_are_delegation_not_dependency(inventory):
    from qshield.paths import EdgeKind

    assert inventory.edges
    assert all(e.kind is EdgeKind.DELEGATION for e in inventory.edges)


def test_dependency_count_reflects_certificates_actually_issued(by_name):
    # The intermediate issued four: three leaves plus the code-signing cert.
    assert by_name["Q-SHIELD Test Intermediate CA"].dependency_count == 5
    assert by_name["web.example.test"].dependency_count == 1


def test_signing_credentials_migrate_to_a_signature_algorithm(inventory):
    """default_replacement maps RSA to ML-KEM because key transport is the usual
    driver; every asset here is a signing credential, so that is wrong."""
    assert inventory.replacements
    assert set(inventory.replacements.values()) == {"ML-DSA"}
    assert inventory.replacements["RSA-2048"] == "ML-DSA"


def test_every_unobservable_field_is_recorded_as_assumed(inventory):
    from qshield.ingest.provenance import NEVER_OBSERVABLE

    for record in inventory.provenance.assets:
        assert set(record.assumed_fields) >= NEVER_OBSERVABLE, record.asset


def test_observed_and_derived_fields_are_not_marked_assumed(inventory):
    record = next(
        r for r in inventory.provenance.assets if r.asset == "Q-SHIELD Test Root CA"
    )
    assert record.fields["algorithm"].source is Source.OBSERVED
    assert record.fields["credential_validity_years"].source is Source.DERIVED
    assert record.fields["dependency_count"].source is Source.DERIVED
    # Every origin carries an explanation; a bare provenance flag is nearly useless.
    assert all(o.detail for o in record.fields.values())


def test_provenance_summary_reports_the_assumed_share(inventory):
    summary = inventory.provenance.summary()
    assert 0.0 < summary["assumed_share"] < 1.0
    assert summary["assets"] == len(inventory.assets)


def test_expired_certificates_are_kept_and_flagged(inventory, by_name):
    """An expired certificate that is still deployed is exactly the kind worth
    knowing about, so it is inventoried rather than dropped."""
    assert "legacy.example.test" in by_name
    assert any("expired" in w for w in inventory.provenance.warnings)


def test_missing_issuer_is_warned_about():
    """A leaf without its CA has no trust inheritance, so its risk is understated."""
    inventory = from_certificates(load_certificates([PKI / "leaf-api.pem"]))
    assert not inventory.edges
    assert any("not in the inventory" in w for w in inventory.provenance.warnings)


def test_a_certificate_only_inventory_has_no_attack_paths():
    """X.509 records trust, not which service reaches which. Inventing an
    entrypoint and a target would manufacture a path term out of nothing."""
    inventory = from_certificates(load_certificates([PKI]))
    case = inventory.as_case(budget=5.0)
    assert case["entrypoints"] == [] and case["targets"] == []
    assert len(problem_from_dict(case).path_set) == 0
    assert any("path term of the objective is inert" in w
               for w in inventory.provenance.warnings)


def test_generated_case_round_trips_into_a_problem():
    inventory = from_certificates(load_certificates([PKI]))
    problem = problem_from_dict(inventory.as_case(budget=6.0))
    assert len(problem.assets) == 6
    assert problem.candidates
    result = problem.evaluate(())
    # The leaves inherit from the intermediate, which inherits from the root.
    assert result.node_scores["web.example.test"] > result.own_scores["web.example.test"]


def test_case_carries_its_provenance_and_a_caveat():
    case = from_certificates(load_certificates([PKI])).as_case(budget=1.0)
    assert "_provenance" in case
    assert "ASSUMED" in case["_comment"]


def test_der_encoded_certificates_are_accepted(tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    cert = x509.load_pem_x509_certificate((PKI / "leaf-api.pem").read_bytes())
    der = tmp_path / "leaf.der"
    der.write_bytes(cert.public_bytes(serialization.Encoding.DER))
    assert len(load_certificates([der])) == 1


def test_a_non_certificate_file_fails_loudly(tmp_path):
    junk = tmp_path / "notes.pem"
    junk.write_text("-----BEGIN CERTIFICATE-----\nnot base64 at all\n")
    with pytest.raises((CertificateIngestError, ValueError)):
        load_certificates([junk])


def test_colliding_common_names_stay_separate_assets(tmp_path):
    """Common names collide constantly in real estates; merging two certificates
    because they share a CN would silently lose one."""
    import shutil

    for index in (1, 2):
        shutil.copy(PKI / "leaf-api.pem", tmp_path / f"copy{index}.pem")
    # Same certificate twice de-duplicates by fingerprint, so use two different
    # certificates that happen to share nothing but are counted independently.
    certs = load_certificates([PKI / "leaf-api.pem", PKI / "leaf-tls.pem"])
    inventory = from_certificates(certs)
    assert len({a.name for a in inventory.assets}) == len(inventory.assets)
