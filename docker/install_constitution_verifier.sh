#!/bin/sh
# Install GitHub's offline publisher-attestation verifier in agent images.
set -eu

keyring=/etc/apt/keyrings/githubcli-archive-keyring.gpg
mkdir -p /etc/apt/keyrings
curl --fail --silent --show-error --location https://cli.github.com/packages/githubcli-archive-keyring.gpg -o "$keyring"
# Public keyring checksum published at github.com/cli/cli/blob/trunk/docs/install_linux.md.
echo "6084d5d7bd8e288441e0e94fc6275570895da18e6751f70f057485dc2d1a811b  $keyring" | sha256sum --check --status
chmod 644 "$keyring"
echo "deb [arch=$(dpkg --print-architecture) signed-by=$keyring] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list
apt-get update
apt-get install -y --no-install-recommends gh
gh attestation verify --help > /dev/null
rm -rf /var/lib/apt/lists/*
