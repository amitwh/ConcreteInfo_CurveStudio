#!/usr/bin/env bash
# Restores a backup made by backup.sh. DESTRUCTIVE: drops and recreates the database, so
# everything currently in the app (accounts, datasets, fits) is replaced by the backup.
#
#   ./restore.sh backups/curvestudio-20260101-053000.sql.gz
#
# The app is restarted afterwards; its schema migrations re-run as a no-op on the restored
# database (the dump already carries the schema).

set -euo pipefail

cd "$(dirname "$0")"

if [ $# -ne 1 ]; then
  echo "usage: $0 <backup.sql.gz>" >&2
  exit 2
fi
file=$1
if [ ! -f "$file" ]; then
  echo "restore: no such file: $file" >&2
  exit 1
fi

printf 'This REPLACES the whole curvestudio database with %s.\n' "$file"
printf 'Type "restore" to continue: '
read -r answer
if [ "$answer" != "restore" ]; then
  echo "restore: aborted"
  exit 1
fi

# Stop the app first: it holds open database connections and dropdb refuses while it runs.
docker compose stop app >/dev/null

docker compose exec -T db sh -c 'exec dropdb -U "$POSTGRES_USER" --if-exists "$POSTGRES_DB"'
docker compose exec -T db sh -c 'exec createdb -U "$POSTGRES_USER" "$POSTGRES_DB"'
gunzip -c "$file" | docker compose exec -T db sh -c 'exec psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -q'
docker compose start app >/dev/null

echo "restore: database restored from $file and the app started again"
