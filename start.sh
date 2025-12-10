#!/bin/bash
# Quick start script for vLLM CPU deployment

set -e

echo "================================================"
echo "vLLM CPU Deployment - Quick Start"
echo "================================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Using default configuration from docker-compose.yml"
    echo ""
fi

# Build and start services
echo "Starting vLLM service..."
echo "This may take a while on first run (model download)"
echo ""

docker compose up -d

echo ""
echo "Waiting for service to be ready..."
echo "This can take 5-10 minutes on first startup..."
echo ""

# Wait for health check
MAX_RETRIES=60
RETRY_COUNT=0
SLEEP_TIME=10

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -sf http://localhost:8009/health > /dev/null 2>&1; then
        echo ""
        echo "✓ vLLM service is ready!"
        echo ""
        echo "================================================"
        echo "Service Information"
        echo "================================================"
        echo "API URL: http://localhost:8009"
        echo "Health: http://localhost:8009/health"
        echo "Models: http://localhost:8009/v1/models"
        echo ""
        echo "View logs:"
        echo "  docker compose logs -f vllm-cpu"
        echo ""
        echo "Test the API:"
        echo "  python test_vllm.py"
        echo ""
        echo "Stop the service:"
        echo "  docker compose down"
        echo ""
        exit 0
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep $SLEEP_TIME
done

echo ""
echo "⚠️  Service did not become ready in time"
echo ""
echo "Check logs with:"
echo "  docker compose logs vllm-cpu"
echo ""
echo "Common issues:"
echo "  - Insufficient memory (check Docker Desktop settings)"
echo "  - Model download in progress (wait longer)"
echo "  - Port 8000 already in use (change VLLM_PORT in .env)"
echo ""

exit 1
