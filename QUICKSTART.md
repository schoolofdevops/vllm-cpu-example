# Quick Start Guide

## 5-Minute Setup

### 1. Start the Service
```bash
# Option A: Using the start script
./start.sh

# Option B: Using make
make start

# Option C: Using docker compose directly
docker compose up -d
```

### 2. Test the API
```bash
# Wait for service to be ready (check health)
make health

# Run full test suite
python test_vllm.py

# Quick curl test
curl http://localhost:8009/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "prompt": "Hello, my name is",
    "max_tokens": 50
  }'
```

## Common Commands

```bash
# View logs
make logs
# or: docker compose logs -f vllm-cpu

# Check status
make status
# or: docker compose ps

# Stop service
make stop
# or: docker compose down

# Restart service
make restart
# or: docker compose restart

# Clean everything
make clean
# or: docker compose down -v
```

## Configuration Presets

### Minimal (2GB RAM)
Edit `.env` and uncomment:
```env
MODEL_NAME=HuggingFaceTB/SmolLM2-135M-Instruct
MEMORY_LIMIT=4G
```

### Balanced (4GB RAM) - Default
Already configured in `.env`

### Maximum Quality (10GB RAM)
Edit `.env` and uncomment:
```env
MODEL_NAME=HuggingFaceTB/SmolLM2-1.7B-Instruct
MEMORY_LIMIT=12G
```

After changing `.env`, restart:
```bash
docker compose down && docker compose up -d
```

## Troubleshooting

### Service won't start
```bash
# Check Docker is running
docker info

# View detailed logs
docker compose logs vllm-cpu

# Check Docker Desktop memory settings
# Ensure at least 4GB allocated to Docker
```

### Out of memory
```bash
# Stop service
docker compose down

# Edit .env - use smaller model
MODEL_NAME=HuggingFaceTB/SmolLM2-135M-Instruct
MEMORY_LIMIT=4G

# Restart
docker compose up -d
```

### Slow responses
```bash
# Edit .env - increase threads
OMP_THREADS=4  # or 6 for Apple Silicon
CPU_LIMIT=6.0

# Restart
docker compose restart
```

## API Examples

### Python with OpenAI Client
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8009/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### curl
```bash
# Completion
curl http://localhost:8009/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "HuggingFaceTB/SmolLM2-360M-Instruct", "prompt": "Hello", "max_tokens": 50}'

# Chat
curl http://localhost:8009/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "HuggingFaceTB/SmolLM2-360M-Instruct", "messages": [{"role": "user", "content": "Hi"}]}'
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore different models on [HuggingFace](https://huggingface.co/HuggingFaceTB)
- Integrate with your applications using the OpenAI-compatible API
- Tune performance settings in `.env` for your hardware
