# ============================================================
# Stage 1: Pre-bake the Airlines demo database
# ============================================================
FROM postgres:16 AS data-loader

ENV POSTGRES_PASSWORD=srelab

COPY server/data/demo-big-en-20170815.sql /tmp/demo.sql

# Start PostgreSQL with bulk-load tuning, load the dump, stop cleanly.
RUN set -e && \
    docker-entrypoint.sh postgres \
        -c wal_level=minimal \
        -c max_wal_senders=0 \
        -c synchronous_commit=off \
        -c fsync=off \
        -c full_page_writes=off \
        -c checkpoint_timeout=30min \
        -c maintenance_work_mem=512MB \
        & \
    sleep 5 && \
    until pg_isready -U postgres; do sleep 1; done && \
    createdb -U postgres demo && \
    psql -U postgres -d demo -f /tmp/demo.sql 2>&1 | tail -5 || true && \
    psql -U postgres -d demo -c "ALTER DATABASE demo SET search_path TO bookings, public;" && \
    su postgres -c "pg_ctl stop -m fast -w -D /var/lib/postgresql/data" && \
    rm /tmp/demo.sql && \
    # Move data to a non-VOLUME path so it persists across stages
    cp -a /var/lib/postgresql/data /var/lib/postgresql/prebaked

# ============================================================
# Stage 2: Final image with pre-loaded data + Python app
# ============================================================
FROM postgres:16

# Install Python and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv curl && \
    rm -rf /var/lib/apt/lists/*

# Create venv and install Python deps
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy pyproject.toml first for better layer caching
COPY pyproject.toml /app/pyproject.toml
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    "openenv-core>=0.2.0" \
    fastapi \
    "uvicorn[standard]" \
    psycopg2-binary \
    "pydantic>=2.0" \
    openai \
    "gradio>=4.0"

# Copy application code (build context = sqlab/)
COPY . /app/sqlab/

# Make the package installable/importable
RUN pip install --no-cache-dir -e /app/

# Store pre-baked data in a non-VOLUME path (Docker VOLUME at
# /var/lib/postgresql/data prevents build-time persistence).
# start.sh will copy this to the real data dir on first boot.
COPY --from=data-loader /var/lib/postgresql/prebaked /var/lib/postgresql/prebaked

# OpenEnv's from_docker_image() maps to container port 8000
# HF Spaces uses app_port in README metadata to route to 8000
EXPOSE 8000

ENV PORT=8000

# Postgres env vars
ENV POSTGRES_PASSWORD=srelab
ENV DB_HOST=localhost
ENV DB_PORT=5432
ENV DB_NAME=demo
ENV DB_USER=postgres
ENV DB_PASSWORD=srelab

# Copy and prepare start script
COPY server/start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
