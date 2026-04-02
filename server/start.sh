#!/bin/bash
set -e

PGDATA=/var/lib/postgresql/data
PREBAKED=/var/lib/postgresql/prebaked

# ── Phase 1: Restore pre-baked data if available ──
# Docker VOLUME at $PGDATA prevents build-time persistence, so we store
# pre-baked data at $PREBAKED and copy it to $PGDATA on first boot.
if [ -d "$PREBAKED/base" ] && [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "=== Restoring pre-baked database ==="
    cp -a "$PREBAKED/." "$PGDATA/"
    chown -R postgres:postgres "$PGDATA"
    chmod 0700 "$PGDATA"
    echo "=== Pre-baked data restored ==="
fi

# ── Phase 2: Start PostgreSQL ──
echo "=== SQLab: Starting PostgreSQL ==="
docker-entrypoint.sh postgres &
PG_PID=$!

echo "=== Waiting for PostgreSQL to be ready ==="
sleep 2
until pg_isready -U postgres -h localhost 2>/dev/null; do
    sleep 1
done
sleep 1
until pg_isready -U postgres -h localhost 2>/dev/null; do
    sleep 1
done
echo "=== PostgreSQL is ready ==="

# Safety net: if data wasn't pre-baked, load it now (idempotent)
createdb -U postgres demo 2>/dev/null || true
LOADED=$(psql -U postgres -d demo -tAc "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'bookings'" 2>/dev/null || echo "")
if [ "$LOADED" != "1" ]; then
    echo "=== Data not pre-baked, loading SQL dump ==="
    psql -U postgres -d demo -f /app/data/demo-big-en-20170815.sql 2>&1 | tail -5 || true
    psql -U postgres -d demo -c "ALTER DATABASE demo SET search_path TO bookings, public;" 2>/dev/null || true
    echo "=== SQL dump loading complete ==="
else
    echo "=== Pre-baked data detected, skipping load ==="
fi

# ── Phase 3: Start FastAPI ──
echo "=== Starting FastAPI server on port ${PORT:-8000} ==="
exec /app/venv/bin/uvicorn sqlab.server.app:app --host 0.0.0.0 --port ${PORT:-8000}
