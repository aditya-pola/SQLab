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

# Copy Airlines demo SQL to /app/data/ (loaded by start.sh, NOT initdb)
COPY server/data/demo-big-en-20170815.sql /app/data/demo-big-en-20170815.sql

# Expose FastAPI port
EXPOSE 8000

# Postgres env vars — don't set POSTGRES_DB so initdb creates only the default 'postgres' db
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
