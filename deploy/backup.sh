#!/usr/bin/env bash
# Dumps the CurveStudio Postgres database through the running compose stack into a
# timestamped, gzip-compressed SQL file. Safe to run while the app is up.
#
#   ./backup.sh                 # writes deploy/backups/curvestudio-YYYYMMDD-HHMMSS.sql.gz
#   ./backup.sh /var/backups    # or any other existing directory
#
# Also back up deploy/.env (the JWT keys) yourself - it is not part of this dump.

set -euo pipefail

cd "$(dirname "$0")"

out_dir=${1:-backups}
stamp=$(TZ=Asia/Kolkata date +%Y%m%d-%H%M%S)
file="$out_dir/curvestudio-$stamp.sql.gz"

if [ -e "$file" ]; then
  echo "backup: refusing to overwrite $file" >&2
  exit 1
fi
mkdir -p "$out_dir"

# The credentials never touch this script: the db container knows its own POSTGRES_* values.
docker compose exec -T db sh -c 'exec pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' | gzip -c > "$file"

echo "backup: wrote $file ($(du -h "$file" | cut -f1))"
