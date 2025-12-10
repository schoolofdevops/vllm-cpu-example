# vLLM CPU-Optimized Deployment for macOS

A production-ready Docker Compose setup for running vLLM-based LLM inference on CPU-only systems, optimized for macOS with minimal footprint.

## Overview

This setup demonstrates how to:
- Deploy vLLM for CPU-only inference on macOS
- Serve small, efficient models (SmolLM2 family)
- Optimize resource usage for local development
- Build custom vLLM images with critical patches

## Features

- **CPU-Optimized**: Patched vLLM with NUMA node handling for containerized environments
- **Small Footprint**: Configurable memory limits and model sizes
- **macOS Compatible**: Thread tuning for Apple Silicon (M1/M2) and Intel Macs
- **Production Ready**: Health checks, automatic restarts, and resource limits
- **Easy Configuration**: Environment-based setup with presets
- **Interactive Chatbot**: Gradio-based web interface included

## Workshop

📚 **Teaching a Workshop?** Check out our comprehensive workshop guide:
- [WORKSHOP.md](WORKSHOP.md) - Complete 2-3 hour workshop curriculum
- [WORKSHOP_SETUP.md](WORKSHOP_SETUP.md) - Pre-workshop setup checklist for participants

The workshop covers:
- Comparing default vs. optimized vLLM images
- Understanding Dockerfile optimization techniques
- Building and deploying with Docker Compose
- Creating an interactive chatbot with Gradio
- Performance tuning and optimization experiments

## Quick Start

### Prerequisites

- Docker Desktop for Mac (4.x or later)
- At least 4GB free RAM (8GB recommended)
- 10GB free disk space

### 1. Clone/Navigate to Directory

```bash
cd /path/to/vllm-cpu
```

### 2. Configure Settings (Optional)

Edit `.env` to customize model and resource limits:

```bash
# Use the default balanced preset (360M model)
# Or uncomment one of the presets at the bottom of .env
```

### 3. Start the Service

```bash
# Build and start vLLM
docker compose up -d

# View logs
docker compose logs -f vllm-cpu

# Wait for model download and initialization (first run may take 5-10 minutes)
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8009/health

# List available models
curl http://localhost:8009/v1/models

# Generate text
curl http://localhost:8009/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "prompt": "What is the capital of France?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Configuration Presets

### Minimal Footprint (~2GB RAM)
```env
MODEL_NAME=HuggingFaceTB/SmolLM2-135M-Instruct
MAX_MODEL_LEN=1024
MEMORY_LIMIT=4G
```

### Balanced (Default, ~4GB RAM)
```env
MODEL_NAME=HuggingFaceTB/SmolLM2-360M-Instruct
MAX_MODEL_LEN=2048
MEMORY_LIMIT=8G
```

### Maximum Quality (~10GB RAM)
```env
MODEL_NAME=HuggingFaceTB/SmolLM2-1.7B-Instruct
MAX_MODEL_LEN=4096
MEMORY_LIMIT=12G
```

## Architecture

### Dockerfile Optimizations

1. **Base Image**: `openeuler/vllm-cpu:0.9.1-oe2403lts`
   - Pre-built vLLM with CPU optimizations
   - OpenEuler Linux for stability

2. **NUMA Patch**: Fixes division-by-zero on systems without NUMA nodes
   ```dockerfile
   RUN sed -i 's/cpu_count_per_numa = cpu_count // numa_size/\
       cpu_count_per_numa = cpu_count // numa_size if numa_size > 0 else cpu_count/g' \
       /workspace/vllm/vllm/worker/cpu_worker.py
   ```

3. **Environment Tuning**:
   - `VLLM_CPU_KVCACHE_SPACE=1`: Limited key-value cache for memory efficiency
   - `OMP_NUM_THREADS=2`: Controlled parallelism to avoid CPU thrashing
   - `OPENBLAS_NUM_THREADS=1`: Single-threaded BLAS operations
   - `MKL_NUM_THREADS=1`: Single-threaded Intel MKL

### Resource Limits

Docker Compose applies CPU and memory limits to prevent system overload:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'        # Maximum CPU cores
      memory: 8G         # Maximum RAM
    reservations:
      cpus: '2.0'        # Guaranteed CPU cores
      memory: 4G         # Guaranteed RAM
```

## Performance Tuning

### For Apple Silicon (M1/M2/M3)

```env
OMP_THREADS=4          # M1/M2 have 8+ cores
CPU_LIMIT=6.0          # Use more cores
MEMORY_LIMIT=12G       # If you have 16GB+ RAM
```

### For Intel Macs

```env
OMP_THREADS=2          # Conservative threading
CPU_LIMIT=4.0          # Moderate CPU usage
MEMORY_LIMIT=8G        # Standard allocation
```

### Memory Optimization

If running low on memory:

1. Reduce `MAX_MODEL_LEN` (limits context window)
2. Reduce `MAX_NUM_SEQS` (limits concurrent requests)
3. Reduce `KVCACHE_SPACE` (limits cached tokens)
4. Switch to a smaller model (135M instead of 360M)

### CPU Optimization

For better responsiveness:

1. Increase `OMP_THREADS` (if you have CPU headroom)
2. Increase `CPU_LIMIT` in .env
3. Close other resource-intensive applications

## API Usage Examples

### Using Python

See `test_vllm.py` for a complete example:

```bash
python test_vllm.py
```

### Using curl

#### Chat Completion
```bash
curl http://localhost:8009/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "messages": [
      {"role": "user", "content": "Explain Docker in one sentence."}
    ],
    "max_tokens": 50
  }'
```

#### Streaming Response
```bash
curl http://localhost:8009/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "prompt": "Write a haiku about containers:",
    "max_tokens": 50,
    "stream": true
  }'
```

### Using OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8009/v1",
    api_key="dummy"  # vLLM doesn't require authentication
)

response = client.chat.completions.create(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    messages=[
        {"role": "user", "content": "What is Docker?"}
    ]
)

print(response.choices[0].message.content)
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker compose logs vllm-cpu

# Common issues:
# 1. Insufficient memory - reduce MEMORY_LIMIT in .env
# 2. Model download failed - check internet connection
# 3. Port conflict - change VLLM_PORT in .env
```

### Out of memory errors

```bash
# Stop the service
docker compose down

# Edit .env and reduce memory usage:
# - Switch to SmolLM2-135M-Instruct
# - Set MAX_MODEL_LEN=1024
# - Set MEMORY_LIMIT=4G

# Restart
docker compose up -d
```

### Slow inference

```bash
# Check CPU usage
docker stats vllm-smollm2

# Increase thread count in .env:
OMP_THREADS=4  # Or higher based on your CPU
```

### Model download stuck

```bash
# Download can take 5-10 minutes on first run
# Monitor progress:
docker compose logs -f vllm-cpu

# If truly stuck, restart:
docker compose restart vllm-cpu
```

## Advanced Usage

### Using Local Models

Mount a local model directory:

```yaml
volumes:
  - ./models:/workspace/models:ro
```

Then set:
```env
MODEL_NAME=/workspace/models/my-model
```

### Adding Web UI

Uncomment the `webui` service in `docker-compose.yml`:

```bash
docker compose up -d
# Access UI at http://localhost:3000
```

### Multi-Model Setup

Create additional service definitions in `docker-compose.yml` with different ports and models.

### Using with LangChain

```python
from langchain.llms import OpenAI

llm = OpenAI(
    openai_api_base="http://localhost:8009/v1",
    openai_api_key="dummy",
    model_name="HuggingFaceTB/SmolLM2-360M-Instruct"
)

response = llm("Explain vLLM in one sentence.")
print(response)
```

## Maintenance

### Update vLLM

```bash
# Pull latest base image
docker compose pull

# Rebuild with no cache
docker compose build --no-cache

# Restart services
docker compose up -d
```

### Clean Up

```bash
# Stop and remove containers
docker compose down

# Remove volumes (clears cached models)
docker compose down -v

# Remove built images
docker rmi vllm-cpu-optimized:latest
```

## Resource Requirements

| Model | Disk Space | RAM (Min) | RAM (Recommended) |
|-------|-----------|-----------|-------------------|
| SmolLM2-135M | ~500MB | 2GB | 4GB |
| SmolLM2-360M | ~1.3GB | 4GB | 8GB |
| SmolLM2-1.7B | ~6.5GB | 8GB | 12GB |

## License

This deployment configuration is provided as-is. vLLM and the models have their own licenses.

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [SmolLM2 Models](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9)
- [OpenAI API Compatibility](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
