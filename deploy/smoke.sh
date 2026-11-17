#!/usr/bin/env bash
# End-to-end smoke test against a running CurveStudio deployment. Works with any base URL -
# a local docker run, the self-host compose stack, or the hosted instance.
#
#   deploy/smoke.sh http://127.0.0.1:8000
#   deploy/smoke.sh https://localhost --insecure   # fresh compose stack, self-signed cert
#
# Checks: /api/v1/health (200, database ok), the SPA is served (index.html and one hashed
# asset), the model catalogue is installed (>= 17 models), then runs a real flow: register a
# throwaway user -> login -> upload a small CSV -> fit a degree-2 polynomial -> wait for the
# job -> verify it converged with the right coefficients -> delete the user again.
#
# Exits non-zero on the first failure (the throwaway account is still cleaned up on the way
# out). Needs curl and python3; registration must be enabled on the target server. --insecure
# skips TLS certificate verification (only for a brand-new stack whose certificate has not
# been issued/trusted yet).

set -euo pipefail

base=${1:?usage: deploy/smoke.sh <base-url> [--insecure]}
base=${base%/}
insecure=()
if [ "${2:-}" = "--insecure" ]; then
  insecure=(-k)
elif [ -n "${2:-}" ]; then
  echo "usage: deploy/smoke.sh <base-url> [--insecure]" >&2
  exit 2
fi
tmp=$(mktemp -d)
token=""

step() { printf 'smoke: %s\n' "$*"; }
fail() { printf 'smoke: FAIL: %s\n' "$*" >&2; exit 1; }

cleanup() {
  if [ -n "$token" ]; then
    if ! curl -sS "${insecure[@]}" -o /dev/null -X DELETE \
      -H "Authorization: Bearer $token" "$base/api/v1/me"; then
      printf 'smoke: warning: could not delete the throwaway account\n' >&2
    fi
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT

for needed in curl python3; do
  command -v "$needed" >/dev/null 2>&1 || fail "$needed is required"
done

# GET/POST helpers: response body lands in $tmp/body, the HTTP status on stdout.
get() { curl -sS "${insecure[@]}" -o "$tmp/body" -w '%{http_code}' "$base$1"; }
auth_get() {
  curl -sS "${insecure[@]}" -o "$tmp/body" -w '%{http_code}' -H "Authorization: Bearer $token" "$base$1"
}
post() { # path json-body
  curl -sS "${insecure[@]}" -o "$tmp/body" -w '%{http_code}' -X POST \
    -H "Authorization: Bearer $token" -H "Content-Type: application/json" -d "$2" "$base$1"
}
field() { # key <- $tmp/body
  python3 -c '
import json, sys
try:
    data = json.load(open(sys.argv[2]))
except Exception as exc:
    sys.exit(f"smoke: response is not JSON: {exc}")
try:
    print(data[sys.argv[1]])
except (KeyError, TypeError):
    sys.exit(f"smoke: no field {sys.argv[1]!r} in response: {sys.argv[2]}")
' "$1" "$tmp/body"
}

step "health"
status=$(get /api/v1/health)
[ "$status" = 200 ] || fail "/api/v1/health returned $status: $(cat "$tmp/body")"
[ "$(field status)" = ok ] || fail "health status is not ok: $(cat "$tmp/body")"
[ "$(field database)" = ok ] || fail "health cannot reach its database: $(cat "$tmp/body")"

step "SPA is served"
status=$(get /)
[ "$status" = 200 ] || fail "/ returned $status"
grep -qi '<!doctype html' "$tmp/body" || fail "/ did not return the SPA's index.html"
asset=$(grep -o 'src="/assets/[^"]*\.js"' "$tmp/body" | head -1 | sed 's/^src="//; s/"$//')
[ -n "$asset" ] || fail "index.html references no /assets/*.js bundle"
status=$(get "$asset")
[ "$status" = 200 ] && [ -s "$tmp/body" ] || fail "asset $asset returned $status"

step "model catalogue"
status=$(get /api/v1/models)
[ "$status" = 200 ] || fail "/api/v1/models returned $status"
models=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$tmp/body")
[ "$models" -ge 17 ] || fail "only $models models listed - the model library packages are not installed"

step "register a throwaway account"
email="smoke-$(date +%s)-$$-${RANDOM}@example.invalid"
password="smoke-${RANDOM}-${RANDOM}-${RANDOM}"
status=$(curl -sS "${insecure[@]}" -o "$tmp/body" -w '%{http_code}' -X POST -H "Content-Type: application/json" \
  -d "{\"email\":\"$email\",\"password\":\"$password\",\"name\":\"Smoke Test\"}" "$base/api/v1/auth/register")
if [ "$status" != 201 ]; then
  [ "$status" = 403 ] && fail "registration is disabled on this server (CURVESTUDIO_REGISTRATION_ENABLED)"
  fail "register returned $status: $(cat "$tmp/body")"
fi

step "login"
status=$(curl -sS "${insecure[@]}" -o "$tmp/body" -w '%{http_code}' -X POST -H "Content-Type: application/json" \
  -d "{\"email\":\"$email\",\"password\":\"$password\"}" "$base/api/v1/auth/login")
[ "$status" = 200 ] || fail "login returned $status: $(cat "$tmp/body")"
token=$(field access_token)
[ -n "$token" ] || fail "login returned no access token"

step "upload a dataset"
# y = 2 + 3x - 0.5x^2 plus a tiny deterministic wobble, so the fit has something to do.
python3 - "$tmp/data.csv" <<'PY'
import sys
with open(sys.argv[1], "w", encoding="ascii") as out:
    out.write("x,y\n")
    for i in range(30):
        x = i
        y = 2.0 + 3.0 * x - 0.5 * x * x + (i % 3) * 1e-6
        out.write(f"{x},{y!r}\n")
PY
status=$(curl -sS "${insecure[@]}" -o "$tmp/body" -w '%{http_code}' -H "Authorization: Bearer $token" \
  -F "file=@$tmp/data.csv;type=text/csv" -F "name=smoke" "$base/api/v1/datasets")
[ "$status" = 201 ] || fail "dataset upload returned $status: $(cat "$tmp/body")"
dataset_id=$(field id)

step "queue a degree-2 polynomial fit"
status=$(post "/api/v1/datasets/$dataset_id/fits" \
  '{"model":{"kind":"polynomial","degree":2},"seed":0}')
[ "$status" = 202 ] || fail "fit returned $status: $(cat "$tmp/body")"
job_id=$(field job_id)

step "wait for the fit job"
job=""
for _ in $(seq 1 90); do
  status=$(auth_get "/api/v1/jobs/$job_id")
  [ "$status" = 200 ] || fail "job polling returned $status: $(cat "$tmp/body")"
  job=$(field status)
  case $job in
    queued | running) sleep 2 ;;
    succeeded | failed | cancelled) break ;;
    *) fail "unknown job status: $job" ;;
  esac
done
[ "$job" = succeeded ] || fail "fit job ended as $job: $(cat "$tmp/body")"

step "check the fit result"
status=$(auth_get "/api/v1/datasets/$dataset_id/fits")
[ "$status" = 200 ] || fail "listing fits returned $status"
fit_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[0]["id"])' "$tmp/body")
status=$(auth_get "/api/v1/fits/$fit_id")
[ "$status" = 200 ] || fail "fetching the fit returned $status"
python3 - "$tmp/body" <<'PY'
import json, sys

bundle = json.load(open(sys.argv[1]))
result = bundle["result"]
assert result["status"] == "converged", f"fit did not converge: {result['status']}"
params = {p["name"]: p["value"] for p in result["parameters"]}
for name, truth in (("b0", 2.0), ("b1", 3.0), ("b2", -0.5)):
    got = params[name]
    assert abs(got - truth) < 0.05, f"{name} is {got}, expected about {truth}"
PY

step "delete the throwaway account"
status=$(curl -sS "${insecure[@]}" -o "$tmp/body" -w '%{http_code}' -X DELETE \
  -H "Authorization: Bearer $token" "$base/api/v1/me")
[ "$status" = 204 ] || fail "DELETE /api/v1/me returned $status: $(cat "$tmp/body")"
token=""

printf 'smoke: OK (%s)\n' "$base"
