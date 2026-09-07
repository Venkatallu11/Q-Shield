"""Emit an already-expired leaf certificate signed by the fixture intermediate.

Split out of make_fixtures.sh because `openssl x509 -req` only learned
-not_before/-not_after in 3.2 and the fixtures must be rebuildable on 3.0.
Run from inside tests/fixtures/pki; see scripts/make_fixtures.sh.
"""

import datetime
import pathlib

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

here = pathlib.Path.cwd()
issuer_cert = x509.load_pem_x509_certificate((here / "intermediate-ca.pem").read_bytes())
issuer_key = serialization.load_pem_private_key(
    (here / "intermediate-ca.key").read_bytes(), password=None
)

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
subject = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Q-SHIELD Test"),
    x509.NameAttribute(NameOID.COMMON_NAME, "legacy.example.test"),
])
cert = (
    x509.CertificateBuilder()
    .subject_name(subject)
    .issuer_name(issuer_cert.subject)
    .public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc))
    .not_valid_after(datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc))
    .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
    .add_extension(
        x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
    )
    .add_extension(
        x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
    )
    .add_extension(
        x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_cert.public_key()),
        critical=False,
    )
    .sign(issuer_key, hashes.SHA256())
)
(here / "leaf-expired.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
