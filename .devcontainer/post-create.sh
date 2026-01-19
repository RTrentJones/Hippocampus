#!/bin/bash
set -e

# Set postgres password for psql commands
export PGPASSWORD=postgres

echo "=== Installing Python dependencies ==="
pip install -e ".[dev]"

echo "=== Waiting for PostgreSQL ==="
until pg_isready -h postgres -U postgres; do
    echo "Waiting for postgres..."
    sleep 1
done

echo "=== Creating extensions ==="
psql -h postgres -U postgres << 'EOF'
CREATE EXTENSION IF NOT EXISTS pg_kafka;
CREATE EXTENSION IF NOT EXISTS vector;
EOF

echo "=== Running migrations ==="
psql -h postgres -U postgres -f /workspace/migrations/001_embeddings.sql

echo "=== Verifying setup ==="
psql -h postgres -U postgres << 'EOF'
\dx
SELECT COUNT(*) AS topics FROM kafka.topics;
SELECT COUNT(*) AS embeddings FROM hippocampus.embeddings;
EOF

echo ""
echo "============================================"
echo "✅ Hippocampus dev environment ready!"
echo "============================================"
echo ""
echo "Commands:"
echo "  hippocampus server     # Start MCP server"
echo "  hippocampus consumer   # Start embedding consumer"
echo "  hippocampus both       # Start both (dev mode)"
echo "  pytest                 # Run tests"
echo ""
echo "Test Kafka connectivity:"
echo "  kcat -L -b postgres:9092"
echo ""
echo "Produce a test message:"
echo "  echo '{\"context\":\"test\",\"reasoning\":\"test\",\"action\":\"test\"}' | \\"
echo "    kcat -P -b postgres:9092 -t decisions.test.001"
echo ""
