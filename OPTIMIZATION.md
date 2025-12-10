# Optimization Guide for macOS

## Understanding the Optimization Strategy

This deployment is optimized for **CPU-only inference on macOS** with a focus on:
1. Minimal memory footprint
2. Efficient CPU utilization
3. Stable performance without thermal throttling
4. Fast startup and model loading

## Architecture-Specific Optimizations

### Apple Silicon (M1/M2/M3)

#### Advantages
- Unified memory architecture (fast CPU-GPU memory)
- High memory bandwidth
- Energy-efficient performance cores
- 8+ CPU cores on most models

#### Recommended Settings
```env
# .env configuration for Apple Silicon
OMP_THREADS=4              # Use 4-6 threads (50-75% of P-cores)
CPU_LIMIT=6.0              # Use most cores
MEMORY_LIMIT=12G           # M1/M2 typically have 16GB+

# For larger models on M1 Max/Ultra with 32GB+
MODEL_NAME=HuggingFaceTB/SmolLM2-1.7B-Instruct
MAX_MODEL_LEN=4096
MEMORY_LIMIT=16G
```

#### Thread Strategy
- M1/M2 have performance (P) and efficiency (E) cores
- Keep `OMP_THREADS` at 50-75% of P-cores
- This prevents E-core scheduling which hurts inference latency
- Example: M1 Pro (8 cores = 6P + 2E) → use `OMP_THREADS=4`

### Intel Macs

#### Challenges
- Separate CPU and system memory
- Higher thermal constraints
- Typically fewer cores (4-8)
- Higher power consumption

#### Recommended Settings
```env
# .env configuration for Intel Macs
OMP_THREADS=2              # Conservative threading
CPU_LIMIT=4.0              # Moderate CPU usage
MEMORY_LIMIT=8G            # Standard allocation

# Use smaller models
MODEL_NAME=HuggingFaceTB/SmolLM2-360M-Instruct
MAX_MODEL_LEN=2048
```

#### Thread Strategy
- Intel CPUs benefit less from high thread counts in inference
- Keep `OMP_THREADS=2-4` to avoid thermal throttling
- Monitor CPU temperature with Activity Monitor
- Lower limits extend laptop battery life

## Memory Optimization

### Understanding Memory Usage

Total memory = Model size + KV Cache + Working memory + Overhead

```
SmolLM2-135M:  ~500MB model + 500MB KV + 1GB working = ~2GB total
SmolLM2-360M:  ~1.3GB model + 1GB KV + 2GB working = ~4GB total
SmolLM2-1.7B:  ~6.5GB model + 2GB KV + 4GB working = ~12GB total
```

### Memory Reduction Strategies

#### 1. Reduce KV Cache
```env
KVCACHE_SPACE=0.5  # From default 1GB
MAX_MODEL_LEN=1024  # Shorter context window
```
**Impact**: Less memory, but can't handle long conversations/documents

#### 2. Reduce Concurrent Sequences
```env
MAX_NUM_SEQS=4  # From default 8
```
**Impact**: Less memory, but fewer parallel requests

#### 3. Use Smaller Model
```env
MODEL_NAME=HuggingFaceTB/SmolLM2-135M-Instruct  # From 360M
```
**Impact**: Less memory, but lower quality outputs

#### 4. Disable Features
```env
# In docker-compose.yml, add to command:
--disable-log-requests
--disable-log-stats
```
**Impact**: Minimal memory savings, but helps with I/O

### Docker Desktop Memory Settings

1. Open Docker Desktop → Settings → Resources
2. Set Memory limit:
   - Minimum: 4GB for SmolLM2-360M
   - Recommended: 8GB for SmolLM2-360M
   - Optimal: 12GB+ for SmolLM2-1.7B
3. Set CPU limit to match your `.env` configuration

## CPU Optimization

### Thread Tuning

The key environment variables:

```env
OMP_NUM_THREADS=2      # OpenMP parallelism (main knob)
OPENBLAS_NUM_THREADS=1 # BLAS operations (keep at 1)
MKL_NUM_THREADS=1      # Intel MKL (keep at 1)
```

#### Finding Your Optimal OMP_THREADS

1. Start with 2 threads:
```bash
OMP_THREADS=2 docker compose up -d
python test_vllm.py  # Note the performance
```

2. Try 4 threads:
```bash
OMP_THREADS=4 docker compose restart
python test_vllm.py  # Compare performance
```

3. Try 6 threads (Apple Silicon only):
```bash
OMP_THREADS=6 docker compose restart
python test_vllm.py  # Compare performance
```

4. Use the value with best latency (not necessarily highest throughput)

#### Why Keep BLAS Threads at 1?

- Multiple BLAS threads compete with OMP threads
- Causes CPU contention and cache thrashing
- Single-threaded BLAS + multi-threaded OMP is more efficient for inference

### CPU Affinity (Advanced)

For even better performance, pin vLLM to specific cores:

```yaml
# docker-compose.yml
services:
  vllm-cpu:
    cpuset: "0-3"  # Use first 4 cores only
```

This prevents OS from moving the process between cores.

## Disk I/O Optimization

### Model Caching

Models are cached in a Docker volume:

```bash
# See cache location
docker volume inspect vllm-cpu_hf-cache

# Pre-download models
docker compose run --rm vllm-cpu \
  python -c "from transformers import AutoModelForCausalLM; \
    AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-360M-Instruct')"
```

### Use Local Models

If you have models already downloaded:

```yaml
# docker-compose.yml
volumes:
  - ./models:/workspace/models:ro
```

```env
# .env
MODEL_NAME=/workspace/models/SmolLM2-360M-Instruct
```

## Network Optimization

### For Local Use Only

```yaml
# docker-compose.yml
ports:
  - "127.0.0.1:8000:8000"  # Only accessible from localhost
```

### For LAN Access

```yaml
# docker-compose.yml
ports:
  - "0.0.0.0:8000:8000"  # Accessible from local network
```

Then access from other devices at `http://<your-mac-ip>:8000`

## Build Optimizations

### Multi-Stage Build (Advanced)

Create a smaller final image:

```dockerfile
# Dockerfile.optimized
FROM openeuler/vllm-cpu:0.9.1-oe2403lts AS base

# Patch
RUN sed -i 's/cpu_count_per_numa = cpu_count \/\/ numa_size/\
    cpu_count_per_numa = cpu_count \/\/ numa_size if numa_size > 0 else cpu_count/g' \
    /workspace/vllm/vllm/worker/cpu_worker.py

# Remove unnecessary files
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV VLLM_TARGET_DEVICE=cpu \
    VLLM_CPU_KVCACHE_SPACE=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
```

### Layer Caching

Build with cache for faster rebuilds:

```bash
# Build with cache
docker compose build

# Build without cache (clean rebuild)
docker compose build --no-cache
```

## Performance Benchmarking

### Measure Baseline Performance

```bash
# Run performance test
python test_vllm.py

# Or use benchmark script
time curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "HuggingFaceTB/SmolLM2-360M-Instruct",
       "prompt": "Test prompt",
       "max_tokens": 100}' > /dev/null
```

### Monitor Resource Usage

```bash
# Container stats
docker stats vllm-smollm2 --no-stream

# Detailed metrics
docker compose exec vllm-cpu top
```

### Expected Performance

On M1 Pro (8-core):
- SmolLM2-135M: ~30-50 tokens/sec
- SmolLM2-360M: ~20-30 tokens/sec
- SmolLM2-1.7B: ~5-10 tokens/sec

On Intel i7 (4-core):
- SmolLM2-135M: ~15-25 tokens/sec
- SmolLM2-360M: ~10-15 tokens/sec
- SmolLM2-1.7B: ~3-5 tokens/sec

## Troubleshooting Performance Issues

### High CPU Usage
```env
# Reduce threads
OMP_THREADS=2
CPU_LIMIT=2.0
```

### High Memory Usage
```env
# Reduce cache and sequences
KVCACHE_SPACE=0.5
MAX_NUM_SEQS=4
MAX_MODEL_LEN=1024
```

### Slow First Request
- Normal: Model loading takes time
- Solution: Keep container running
- Or use health check endpoint to warm up

### Thermal Throttling
- Monitor: Activity Monitor → CPU History
- Reduce `OMP_THREADS` and `CPU_LIMIT`
- Ensure good laptop ventilation
- Consider using smaller model

## Production Recommendations

For production deployments on macOS:

```env
# Stable, production settings
MODEL_NAME=HuggingFaceTB/SmolLM2-360M-Instruct
OMP_THREADS=2
CPU_LIMIT=4.0
CPU_RESERVATION=2.0
MEMORY_LIMIT=8G
MEMORY_RESERVATION=4G
MAX_MODEL_LEN=2048
MAX_NUM_SEQS=8
KVCACHE_SPACE=1
```

These settings provide:
- Predictable performance
- No thermal issues
- Reasonable quality
- Good concurrency
- Memory safety

## Advanced: Custom vLLM Configuration

For expert users, you can pass additional vLLM flags:

```yaml
# docker-compose.yml
command: >
  vllm serve ${MODEL_NAME}
  --host 0.0.0.0
  --port 8000
  --dtype auto
  --max-model-len ${MAX_MODEL_LEN}
  --max-num-seqs ${MAX_NUM_SEQS}
  --gpu-memory-utilization 0.0
  --swap-space 0
  --enforce-eager
  --disable-custom-all-reduce
```

See [vLLM docs](https://docs.vllm.ai/en/latest/models/engine_args.html) for all options.
