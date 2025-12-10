.PHONY: help start stop restart logs build clean test health status

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "vLLM CPU Deployment - Available Commands"
	@echo "========================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

start: ## Start vLLM service
	@echo "Starting vLLM service..."
	@docker compose up -d
	@echo "Service started. Run 'make logs' to view output."
	@echo "Run 'make health' to check when ready."

stop: ## Stop vLLM service
	@echo "Stopping vLLM service..."
	@docker compose down

restart: ## Restart vLLM service
	@echo "Restarting vLLM service..."
	@docker compose restart

logs: ## View service logs
	@docker compose logs -f vllm-cpu

build: ## Build Docker image
	@echo "Building vLLM image..."
	@docker compose build --no-cache

clean: ## Stop and remove containers, volumes, and images
	@echo "Cleaning up..."
	@docker compose down -v
	@docker rmi vllm-cpu-optimized:latest 2>/dev/null || true
	@echo "Cleanup complete."

test: ## Run API tests
	@echo "Running API tests..."
	@python test_vllm.py

health: ## Check service health
	@echo "Checking service health..."
	@curl -f http://localhost:8009/health && echo "✓ Service is healthy" || echo "✗ Service is not ready"

status: ## Show container status
	@docker compose ps

models: ## List available models
	@curl -s http://localhost:8009/v1/models | python -m json.tool

quick-test: ## Quick API test with curl
	@echo "Testing completion API..."
	@curl -s http://localhost:8009/v1/completions \
		-H "Content-Type: application/json" \
		-d '{"model": "HuggingFaceTB/SmolLM2-360M-Instruct", "prompt": "Hello, ", "max_tokens": 20}' \
		| python -m json.tool

stats: ## Show container resource usage
	@docker stats vllm-smollm2 --no-stream

# Development targets
dev-start: ## Start with logs following
	@docker compose up -d && docker compose logs -f vllm-cpu

dev-shell: ## Get shell access to container
	@docker compose exec vllm-cpu /bin/bash

# Configuration presets
preset-minimal: ## Switch to minimal footprint preset
	@echo "Switching to minimal footprint preset..."
	@echo "Edit .env and uncomment the 'PRESET: Minimal Footprint' section"

preset-balanced: ## Switch to balanced preset (default)
	@echo "Using balanced preset (default configuration)"

preset-quality: ## Switch to maximum quality preset
	@echo "Switching to maximum quality preset..."
	@echo "Edit .env and uncomment the 'PRESET: Maximum Quality' section"
