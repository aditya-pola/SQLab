#!/bin/bash
set -e

# ── Phase 1: Let the official entrypoint fully initialise Postgres ──
# The entrypoint starts a temp server, runs initdb, shuts it down,
# then starts Postgres for real. We must NOT interfere during that.
echo "=== SQLab: Starting PostgreSQL via official entrypoint ==="
docker-entrypoint.sh postgres &
PG_PID=$!

# Wait for the REAL Postgres (after entrypoint finishes its init cycle).
# The entrypoint creates a sentinel: /var/lib/postgresql/data/PG_VERSION exists
# once initdb has run. But the safest approach is to wait for pg_isready
# and then check the server has been up for more than 2 seconds (to skip
# the temporary initdb server).
echo "=== Waiting for PostgreSQL to be fully ready ==="
sleep 5  # give the entrypoint time to start its init cycle
until pg_isready -U postgres -h localhost 2>/dev/null; do
    sleep 2
done
# Double-check: wait a bit and verify still ready (not the temp server shutting down)
sleep 3
until pg_isready -U postgres -h localhost 2>/dev/null; do
    sleep 2
done
echo "=== PostgreSQL is ready ==="

# ── Phase 2: Create the demo database and load the SQL dump ──
echo "=== Creating demo database ==="
createdb -U postgres demo 2>/dev/null || echo "Database 'demo' already exists, continuing"

# Check if data already loaded (idempotent: skip if bookings schema exists)
LOADED=$(psql -U postgres -d demo -tAc "SELECT 1 FROM information_schema.schemata WHERE schema_name = 'bookings'" 2>/dev/null || echo "")
if [ "$LOADED" != "1" ]; then
    echo "=== Loading Airlines demo SQL dump (this may take several minutes) ==="
    # The dump contains DROP DATABASE which will fail — that's OK, just continue
    psql -U postgres -d demo -f /app/data/demo-big-en-20170815.sql 2>&1 | tail -20 || true
    echo "=== SQL dump loading complete ==="
else
    echo "=== Data already loaded, skipping ==="
fi

# Set search_path to bookings schema for convenience
psql -U postgres -d demo -c "ALTER DATABASE demo SET search_path TO bookings, public;" 2>/dev/null || true

# ── Phase 3: Start FastAPI ──
echo "=== Starting FastAPI server ==="
exec /app/venv/bin/uvicorn sqlab.server.app:app --host 0.0.0.0 --port 8000
