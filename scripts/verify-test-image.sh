#!/bin/bash
set -e

IMAGE_NAME="hippocampus-postgres:test-latest"

echo "========================================="
echo "Verifying Hippocampus Test PostgreSQL Image"
echo "========================================="
echo "Image: ${IMAGE_NAME}"
echo ""
echo "This script verifies the Docker image was built correctly."
echo "It checks that extension files exist in the filesystem."
echo ""

echo "Starting test container (no volumes, no init scripts)..."
CONTAINER_ID=$(docker run -d \
    -e POSTGRES_PASSWORD=postgres \
    ${IMAGE_NAME} \
    postgres -c shared_preload_libraries='pg_kafka,vector')

echo "Container ID: ${CONTAINER_ID}"
echo ""

echo "Waiting for PostgreSQL to start..."
sleep 8

echo ""
echo "1. Verifying PostgreSQL is running..."
if docker exec $CONTAINER_ID pg_isready -U postgres > /dev/null; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL failed to start"
    docker logs $CONTAINER_ID
    docker stop $CONTAINER_ID > /dev/null 2>&1
    docker rm $CONTAINER_ID > /dev/null 2>&1
    exit 1
fi

echo ""
echo "2. Verifying pg_kafka extension files..."
if docker exec $CONTAINER_ID test -f /usr/share/postgresql/17/extension/pg_kafka.control; then
    echo "✅ pg_kafka.control exists"
else
    echo "❌ pg_kafka.control not found"
    docker stop $CONTAINER_ID > /dev/null 2>&1
    docker rm $CONTAINER_ID > /dev/null 2>&1
    exit 1
fi

echo ""
echo "3. Verifying pg_kafka shared library..."
if docker exec $CONTAINER_ID test -f /usr/lib/postgresql/17/lib/pg_kafka.so; then
    echo "✅ pg_kafka.so exists"
else
    echo "❌ pg_kafka.so not found"
    docker stop $CONTAINER_ID > /dev/null 2>&1
    docker rm $CONTAINER_ID > /dev/null 2>&1
    exit 1
fi

echo ""
echo "4. Verifying vector extension files..."
if docker exec $CONTAINER_ID test -f /usr/share/postgresql/17/extension/vector.control; then
    echo "✅ vector.control exists"
else
    echo "❌ vector.control not found"
    docker stop $CONTAINER_ID > /dev/null 2>&1
    docker rm $CONTAINER_ID > /dev/null 2>&1
    exit 1
fi

echo ""
echo "5. Verifying vector shared library..."
if docker exec $CONTAINER_ID test -f /usr/lib/postgresql/17/lib/vector.so; then
    echo "✅ vector.so exists"
else
    echo "❌ vector.so not found"
    docker stop $CONTAINER_ID > /dev/null 2>&1
    docker rm $CONTAINER_ID > /dev/null 2>&1
    exit 1
fi

echo ""
echo "6. Checking PostgreSQL logs for library loading..."
docker exec $CONTAINER_ID cat /var/lib/postgresql/data/log/*.log 2>/dev/null | grep -i "shared_preload_libraries" || echo "  (Log check skipped - file may not exist yet)"

echo ""
echo "Cleaning up test container..."
docker stop $CONTAINER_ID > /dev/null
docker rm $CONTAINER_ID > /dev/null

echo ""
echo "✅ All verifications passed!"
echo ""
echo "The image was built correctly with:"
echo "  - pg_kafka extension files and shared library"
echo "  - vector extension files and shared library"
echo "  - PostgreSQL 17 with proper shared_preload_libraries support"
echo ""
echo "Next steps:"
echo "  1. Start test environment:   docker-compose -f docker-compose.test.yml up -d"
echo "  2. Run tests:                pytest tests/"
echo "  3. Stop environment:         docker-compose -f docker-compose.test.yml down -v"
echo ""
