#!/usr/bin/env bash
# Regenerate the test PKI in tests/fixtures/pki.
#
# The fixtures are committed so the test suite needs neither openssl nor a
# network. This script exists so they can be rebuilt and audited: a parser
# tested only against certificates it produced itself proves very little, so
# these are real certificates emitted by OpenSSL, signed by a real chain.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=tests/fixtures/pki
rm -rf "$OUT" && mkdir -p "$OUT"
cd "$OUT"

# 20-year RSA-4096 root, self-signed.
openssl req -x509 -newkey rsa:4096 -keyout root-ca.key -out root-ca.pem -days 7300 -nodes \
  -subj "/C=US/O=Q-SHIELD Test/CN=Q-SHIELD Test Root CA" \
  -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign,cRLSign" 2>/dev/null

# 8-year ECDSA P-256 intermediate, signed by the root.
openssl req -newkey ec -pkeyopt ec_paramgen_curve:P-256 -keyout intermediate-ca.key -out intermediate-ca.csr -nodes \
  -subj "/C=US/O=Q-SHIELD Test/CN=Q-SHIELD Test Intermediate CA" 2>/dev/null
openssl x509 -req -in intermediate-ca.csr -CA root-ca.pem -CAkey root-ca.key -CAcreateserial \
  -out intermediate-ca.pem -days 2920 \
  -extfile <(printf "basicConstraints=critical,CA:TRUE,pathlen:0\nkeyUsage=critical,keyCertSign,cRLSign\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid:always\n") 2>/dev/null

# 90-day ECDSA leaf for TLS, signed by the intermediate.
openssl req -newkey ec -pkeyopt ec_paramgen_curve:P-256 -keyout leaf-tls.key -out leaf-tls.csr -nodes \
  -subj "/C=US/O=Q-SHIELD Test/CN=web.example.test" 2>/dev/null
openssl x509 -req -in leaf-tls.csr -CA intermediate-ca.pem -CAkey intermediate-ca.key -CAcreateserial \
  -out leaf-tls.pem -days 90 \
  -extfile <(printf "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=DNS:web.example.test,DNS:www.example.test\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid:always\n") 2>/dev/null

# 1-year RSA-2048 leaf, signed by the intermediate.
openssl req -newkey rsa:2048 -keyout leaf-api.key -out leaf-api.csr -nodes \
  -subj "/C=US/O=Q-SHIELD Test/CN=api.example.test" 2>/dev/null
openssl x509 -req -in leaf-api.csr -CA intermediate-ca.pem -CAkey intermediate-ca.key -CAcreateserial \
  -out leaf-api.pem -days 365 \
  -extfile <(printf "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth,clientAuth\nsubjectAltName=DNS:api.example.test\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid:always\n") 2>/dev/null

# 3-year Ed25519 code-signing certificate -- the NON_REPUDIATION case.
openssl req -newkey ed25519 -keyout code-signing.key -out code-signing.csr -nodes \
  -subj "/C=US/O=Q-SHIELD Test/CN=Q-SHIELD Release Signing" 2>/dev/null
openssl x509 -req -in code-signing.csr -CA intermediate-ca.pem -CAkey intermediate-ca.key -CAcreateserial \
  -out code-signing.pem -days 1095 \
  -extfile <(printf "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=codeSigning\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid:always\n") 2>/dev/null

# An expired leaf, to exercise the already-lapsed path. Built with python
# rather than openssl because `x509 -req` gained -not_before/-not_after only in
# OpenSSL 3.2 and this must work on 3.0; it is still a real X.509 certificate,
# signed by the same intermediate key as the rest of the chain.
python3 ../../../scripts/_make_expired_fixture.py

# A full chain in one file, the shape a server usually deploys.
cat leaf-tls.pem intermediate-ca.pem root-ca.pem > chain-bundle.pem

rm -f ./*.key ./*.csr ./*.srl
echo "fixtures written to $OUT:"
ls -1
