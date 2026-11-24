# Verify historical Kestrel rotation records

Some May 2026 Kestrel key rotations used the SPHINCS+-SHA2-128s-simple implementation in pqcrypto 0.4.0. They carry the archival algorithm label later assigned to FIPS 205 SLH-DSA. The signatures and public keys have the same lengths under both algorithms. The bytes require the verifier that matches their documented origin.

For an edition-one adoption using these records, install the historical verifier in its own environment. Run these commands from the repository root.

```sh
uv venv --python 3.13 "$HOME/.kestrel/verification/sphincs-0.4.0"
export KESTREL_HISTORICAL_SPHINCS_PYTHON="$HOME/.kestrel/verification/sphincs-0.4.0/bin/python"
uv pip install --python "$KESTREL_HISTORICAL_SPHINCS_PYTHON" --only-binary :all: --require-hashes -r scripts/verification/legacy-sphincs/requirements.txt
```

Set this interpreter path in the operator environment used for verification. The signed adoption receipt selects the historical algorithm and pins the archival public key. The verifier takes public data, runs offline with an empty environment, and checks the original signed payload. Existing FIPS 205 operations use the normal runtime. Store the historical rotation record with the adoption evidence so either party can repeat the check.

## Verify the second edition’s publisher

Install the GitHub CLI (`gh`) on the operator host. Its `attestation verify` command must support `--bundle`, `--custom-trusted-root`, `--source-digest`, and `--signer-digest`. Kestrel checks the released archive against the publisher’s main-branch workflow, repository identity, source commit, and recorded SHA-256 digests. The distribution includes the archive, its public attestation, and the independently obtained Sigstore trusted roots. Verification runs offline and requires no GitHub login.

To repeat that check from an installed Kestrel environment, run this command.

```sh
python -c 'from kestrel_sovereign.constitution.publisher import verify_packaged_publisher; print(verify_packaged_publisher("d0949dba4b1969bc11098f64d56d38490ea59ddcbe91c40b7775f5d506408908"))'
```
