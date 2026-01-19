#!/bin/bash
set -e

echo "[*] Stopping Hippocampus test environment..."
docker-compose -f docker-compose.test.yml down -v
echo "[+] Test environment stopped and cleaned"
