#!/bin/bash
set -e

echo "[*] Starting Hippocampus test environment..."

# Check if image exists
if ! docker images | grep -q "hippocampus-postgres.*test-latest"; then
    echo "[*] Test image not found. Building (this takes 5-10 minutes)..."
    ./scripts/build-test-image.sh
fi

# Start services
docker-compose -f docker-compose.test.yml up -d

# Wait for healthy status
echo "[*] Waiting for database to be ready..."
timeout 60 bash -c 'until docker-compose -f docker-compose.test.yml ps | grep "(healthy)"; do
    echo -n "."
    sleep 2
done' || {
    echo ""
    echo "[!] Healthcheck failed. Checking logs..."
    docker-compose -f docker-compose.test.yml logs --tail=20
    exit 1
}

echo ""
echo "[+] Test environment ready!"
echo ""
echo "Run tests with:"
echo "  TEST_DATABASE_URL=postgresql://postgres:postgres@hippocampus-test-db:5432/hippocampus_test pytest tests/"
echo ""
echo "Or source .env.test first:"
echo "  source .env.test && pytest tests/"
