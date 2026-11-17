#!/usr/bin/env bash
# Generates the RS256 key pair CurveStudio signs its JWTs with, and prints ready-to-paste
# .env lines (single-line, \n-escaped PEM). Nothing is written outside a private temp dir.
#
#   cd deploy && ./gen-keys.sh >> .env
#
# Requires openssl. The private key never leaves your terminal; only paste the output into
# deploy/.env, which is git-ignored and excluded from the docker build context.

set -euo pipefail

if [ -t 1 ]; then
  echo "gen-keys: refusing to print a private key to a terminal." >&2
  echo "gen-keys: redirect or append to a file instead:  ./gen-keys.sh >> .env" >&2
  exit 1
fi

umask 077
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT

openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out "$dir/private.pem" 2>/dev/null
openssl pkey -in "$dir/private.pem" -pubout -out "$dir/public.pem"

# \n-escape each PEM onto one line (the API accepts literal \n in these variables).
escape() { sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/\\n/g' "$1"; }

printf 'CURVESTUDIO_JWT_PRIVATE_KEY_PEM=%s\n' "$(escape "$dir/private.pem")"
printf 'CURVESTUDIO_JWT_PUBLIC_KEY_PEM=%s\n' "$(escape "$dir/public.pem")"
