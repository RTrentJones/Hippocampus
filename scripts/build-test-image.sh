#!/bin/bash
set -e

# Build the test PostgreSQL image
# Usage: ./scripts/build-test-image.sh [--no-cache]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="hippocampus-postgres"
IMAGE_TAG="test-latest"
PG_KAFKA_VERSION="${PG_KAFKA_VERSION:-main}"

echo "========================================="
echo "Building Hippocampus Test PostgreSQL Image"
echo "========================================="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "pg_kafka version: ${PG_KAFKA_VERSION}"
echo ""

# Parse arguments
NO_CACHE=""
if [[ "$1" == "--no-cache" ]]; then
    NO_CACHE="--no-cache"
    echo "Building with --no-cache"
fi

# Build the image
docker build \
    ${NO_CACHE} \
    --build-arg PG_KAFKA_VERSION=${PG_KAFKA_VERSION} \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    -f docker/Dockerfile.test-postgres \
    "$PROJECT_ROOT"

echo ""
echo "✅ Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Next steps:"
echo "  1. Verify image:             ./scripts/verify-test-image.sh"
echo "  2. Start test environment:   docker-compose -f docker-compose.test.yml up -d"
echo "  3. Run tests:                pytest tests/"
echo "  4. Stop environment:         docker-compose -f docker-compose.test.yml down"
echo ""
