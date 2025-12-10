# Workshop: Building Optimized vLLM for CPU-Based Local Inference

**Duration**: 2-3 hours
**Level**: Intermediate
**Prerequisites**: Basic Docker knowledge, Python familiarity

## Workshop Overview

Learn how to build and optimize vLLM (Very Large Language Model inference engine) for efficient CPU-based local deployments. You'll compare default vLLM images with optimized versions, understand Dockerfile optimization techniques, and build a complete AI chatbot using Gradio.

### What You'll Learn

1. Compare default vs. optimized vLLM Docker images
2. Understand Dockerfile optimization techniques for CPU inference
3. Build custom vLLM images using Docker Compose
4. Deploy vLLM with OpenAI-compatible API
5. Create an interactive chatbot using Gradio

### What You'll Build

- Optimized vLLM container with ~50% memory savings
- RESTful API server for LLM inference
- Interactive web-based chatbot interface

---

## Part 1: Understanding vLLM and Image Comparison (30 mins)

### What is vLLM?

vLLM is a high-performance inference engine for large language models that supports:
- Fast inference with PagedAttention
- Continuous batching for higher throughput
- OpenAI-compatible API
- CPU and GPU backends

### Default vLLM vs. Optimized Version

#### Pulling the Default Image

```bash
# Pull the official vLLM CPU image
docker pull vllm/vllm-openai:latest

# Inspect the image
docker images vllm/vllm-openai:latest
docker inspect vllm/vllm-openai:latest | grep -A 10 "Env"
```

#### Comparison Table

| Aspect | Default vLLM | Our Optimized Version |
|--------|--------------|----------------------|
| Base Image | Ubuntu/Debian (large) | OpenEuler 24.03 LTS (leaner) |
| NUMA Support | May crash on macOS/VMs | Patched for zero NUMA nodes |
| Thread Config | Auto-detected (variable) | Explicitly tuned (predictable) |
| Memory Usage | ~6-8GB for 360M model | ~4-5GB for 360M model |
| KV Cache | Auto (large) | Limited to 1GB (configurable) |
| CPU Optimization | Generic | macOS-specific tuning |
| Image Size | ~4-6GB | ~3-4GB |
| Startup Time | Variable | Faster (pre-configured) |

#### Key Problems We Solve

**Problem 1: NUMA Node Division by Zero**
```python
# Default vLLM code (crashes on Docker/macOS)
cpu_count_per_numa = cpu_count // numa_size

# What happens on systems with 0 NUMA nodes?
# ZeroDivisionError: integer division or modulo by zero
```

**Problem 2: Uncontrolled Resource Usage**
- Default settings can consume all available CPU/memory
- Causes thermal throttling on laptops
- Poor performance with multiple services

**Problem 3: Inconsistent Performance**
- Thread counts vary by system
- No resource limits
- Unpredictable behavior

### Exercise 1: Run Default vLLM (10 mins)

Try running the default vLLM image and observe the issues:

```bash
# Attempt to run default vLLM on macOS
docker run -p 8010:8000 \
  vllm/vllm-openai:latest \
  --model HuggingFaceTB/SmolLM2-360M-Instruct

# Observe: May crash with NUMA errors on macOS/VM environments
# Check logs for errors
docker logs <container-id>
```

**Expected Issues:**
- NUMA-related crashes on containerized environments
- High memory usage
- Inconsistent CPU utilization

---

## Part 2: Understanding the Optimized Dockerfile (30 mins)

### The Dockerfile Breakdown

Let's examine our optimized Dockerfile line by line:

```dockerfile
FROM openeuler/vllm-cpu:0.9.1-oe2403lts
```

**Why this base image?**
- OpenEuler is optimized for enterprise workloads
- Pre-built vLLM 0.9.1 with CPU optimizations
- Smaller footprint than Ubuntu-based images
- Better BLAS/LAPACK integration

```dockerfile
# Patch the cpu_worker.py to handle zero NUMA nodes
RUN sed -i 's/cpu_count_per_numa = cpu_count \/\/ numa_size/\
    cpu_count_per_numa = cpu_count \/\/ numa_size if numa_size > 0 else cpu_count/g' \
    /workspace/vllm/vllm/worker/cpu_worker.py
```

**What this patch does:**
- Fixes division-by-zero bug in vLLM's CPU worker
- Adds conditional check: `if numa_size > 0 else cpu_count`
- Essential for Docker Desktop, VMs, and cloud containers
- Prevents crashes on systems without NUMA topology

**Before patch:**
```python
cpu_count_per_numa = cpu_count // numa_size  # Crashes if numa_size == 0
```

**After patch:**
```python
cpu_count_per_numa = cpu_count // numa_size if numa_size > 0 else cpu_count
```

```dockerfile
ENV VLLM_TARGET_DEVICE=cpu \
    VLLM_CPU_KVCACHE_SPACE=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
```

**Environment variables explained:**

| Variable | Value | Purpose |
|----------|-------|---------|
| VLLM_TARGET_DEVICE | cpu | Force CPU mode (no CUDA) |
| VLLM_CPU_KVCACHE_SPACE | 1 | Limit key-value cache to 1GB |
| OMP_NUM_THREADS | 2 | OpenMP parallelism (main inference) |
| OPENBLAS_NUM_THREADS | 1 | Single-threaded BLAS operations |
| MKL_NUM_THREADS | 1 | Single-threaded Intel MKL |

**Threading Strategy:**
- Multiple OMP threads (2-4) for inference parallelism
- Single-threaded BLAS/MKL to avoid contention
- This pattern prevents CPU thrashing
- Optimal for CPU-bound inference workloads

### Exercise 2: Analyze the Dockerfile (10 mins)

Open the Dockerfile and answer these questions:

1. What would happen without the NUMA patch on macOS?
2. Why do we set OPENBLAS_NUM_THREADS=1 instead of 4?
3. What's the trade-off of VLLM_CPU_KVCACHE_SPACE=1?
4. How could you further reduce memory usage?

**Discussion Points:**
- Memory vs. performance trade-offs
- CPU threading strategies
- Platform-specific optimizations

---

## Part 3: Docker Compose and Image Building (30 mins)

### Understanding docker-compose.yml

#### Service Definition

```yaml
services:
  vllm-cpu:
    build:
      context: .
      dockerfile: Dockerfile
    image: vllm-cpu-optimized:latest
```

- `build.context`: Directory containing Dockerfile
- `dockerfile`: Which Dockerfile to use
- `image`: Tag for the built image

#### Command Configuration

```yaml
command: >
  vllm serve ${MODEL_NAME:-HuggingFaceTB/SmolLM2-360M-Instruct}
  --host 0.0.0.0
  --port 8000
  --dtype ${DTYPE:-auto}
  --max-model-len ${MAX_MODEL_LEN:-2048}
  --max-num-seqs ${MAX_NUM_SEQS:-8}
```

**Key flags explained:**
- `--host 0.0.0.0`: Listen on all interfaces
- `--port 8000`: Internal container port
- `--dtype auto`: Automatic data type selection
- `--max-model-len 2048`: Maximum sequence length (context window)
- `--max-num-seqs 8`: Concurrent request limit

#### Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
```

**Why resource limits matter:**
- Prevents container from consuming entire system
- Ensures predictable performance
- Required for production deployments
- Helps with cost optimization in cloud

### Exercise 3: Build Your First Image (15 mins)

#### Step 1: Examine the Configuration

```bash
# View the .env file
cat .env

# View the Dockerfile
cat Dockerfile

# View the docker-compose.yml
cat docker-compose.yml
```

#### Step 2: Build the Image

```bash
# Build using docker compose
docker compose build

# Or build with no cache (clean build)
docker compose build --no-cache

# Monitor the build process
# Observe each layer being built
```

#### Step 3: Verify the Build

```bash
# List images
docker images | grep vllm-cpu-optimized

# Inspect the image
docker inspect vllm-cpu-optimized:latest

# Check image size
docker images vllm-cpu-optimized:latest --format "{{.Size}}"
```

#### Step 4: Compare Image Sizes

```bash
# Our optimized image
docker images vllm-cpu-optimized:latest

# Default vLLM (if pulled earlier)
docker images vllm/vllm-openai:latest

# Calculate space savings
```

**Expected Results:**
- Build time: 2-5 minutes
- Image size: ~3-4GB
- All layers cached for future builds

### Exercise 4: Understanding Layers (10 mins)

```bash
# View image history (layers)
docker history vllm-cpu-optimized:latest

# Analyze layer sizes
docker history vllm-cpu-optimized:latest --format "table {{.CreatedBy}}\t{{.Size}}"
```

**Discussion:**
- Which layer is the largest?
- How does Docker caching work?
- What happens when you change the Dockerfile?

---

## Part 4: Deploying vLLM for Local Inference (30 mins)

### Starting the Service

We'll use Docker Compose commands directly so you understand exactly what's happening at each step.

#### Step 1: Start the Container

```bash
# Start in detached mode (runs in background)
docker compose up -d

# What this does:
# 1. Builds the image if not already built
# 2. Creates and starts the container
# 3. Returns control to your terminal
```

#### Step 2: Monitor the Startup

```bash
# Follow the logs in real-time
docker compose logs -f vllm-cpu

# Press Ctrl+C to stop following (container keeps running)
```

#### Step 3: Check Container Status

```bash
# See running containers
docker compose ps

# Expected output:
# NAME              IMAGE                       STATUS        PORTS
# vllm-smollm2      vllm-cpu-optimized:latest   Up 2 minutes  0.0.0.0:8009->8000/tcp
```

#### Step 4: Verify Service Health

```bash
# Check health endpoint
curl http://localhost:8009/health

# Keep trying until you get: {"status":"ok"}
# This may take 5-10 minutes on first run (model download)
```

### Understanding the Startup Process

```
Startup Timeline:
┌─────────────────────────────────────────────────────┐
│ 0s:  Container starts                               │
│ 1s:  vLLM begins initialization                     │
│ 5s:  Checking for model in cache                    │
│ 10s: Downloading model from HuggingFace (if needed) │
│ 3m:  Model download complete (~1.3GB)               │
│ 4m:  Loading model into memory                      │
│ 5m:  Initializing inference engine                  │
│ 6m:  API server ready - accepting requests          │
└─────────────────────────────────────────────────────┘
```

### Exercise 5: Monitor the Startup (10 mins)

```bash
# Start the service
docker compose up -d

# Watch logs in real-time
docker compose logs -f vllm-cpu

# In another terminal, monitor resources
docker stats vllm-smollm2

# Wait for this message:
# "Uvicorn running on http://0.0.0.0:8000"
```

**Observe:**
- Model download progress
- Memory allocation
- CPU usage patterns
- Time to first request

### Testing the API

#### 1. Health Check

```bash
# Check if service is ready
curl http://localhost:8009/health

# Expected response:
# {"status": "ok"}
```

#### 2. List Models

```bash
# See available models
curl http://localhost:8009/v1/models | jq

# Expected response:
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
#       "object": "model",
#       "created": 1234567890,
#       "owned_by": "vllm"
#     }
#   ]
# }
```

#### 3. Generate Text (Completion)

```bash
# Simple completion
curl http://localhost:8009/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq
```

#### 4. Chat Completion

```bash
# Chat-style interaction
curl http://localhost:8009/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain Docker in one sentence."}
    ],
    "max_tokens": 100
  }' | jq
```

### Exercise 6: API Exploration (15 mins)

Try these prompts and observe the responses:

```bash
# 1. Code generation
curl http://localhost:8009/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "prompt": "Write a Python function to calculate fibonacci:",
    "max_tokens": 150
  }' | jq '.choices[0].text'

# 2. Question answering
curl http://localhost:8009/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ]
  }' | jq '.choices[0].message.content'

# 3. Creative writing
curl http://localhost:8009/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "prompt": "Write a haiku about Docker containers:",
    "max_tokens": 50,
    "temperature": 0.9
  }' | jq '.choices[0].text'
```

**Discussion Points:**
- How does temperature affect responses?
- What's the difference between completion and chat APIs?
- When would you use streaming vs. non-streaming?

---

## Part 5: Building a Gradio Chatbot (45 mins)

### What is Gradio?

Gradio is a Python library for building interactive web UIs for machine learning models. We'll use it to create a chatbot interface for our vLLM API.

### The Chatbot Application

The chatbot runs as a container alongside vLLM, providing:
- Clean chat interface
- Message history
- Streaming responses
- Adjustable parameters (temperature, max tokens)
- Error handling

All containerized - no local Python installation needed!

### Exercise 7: Launch the Chatbot (20 mins)

#### Step 1: Start Both Services

```bash
# Start both vLLM and the chatbot
docker compose up -d

# What this does:
# 1. Builds the chatbot image (first time only)
# 2. Starts vLLM service
# 3. Starts chatbot service (waits for vLLM)
```

#### Step 2: Monitor the Startup

```bash
# View logs from both services
docker compose logs -f

# Or view just the chatbot:
docker compose logs -f chatbot

# Wait for: "Running on local URL: http://0.0.0.0:7860"
```

#### Step 3: Verify Both Services

```bash
# Check vLLM health
curl http://localhost:8009/health

# Check container status
docker compose ps

# Expected output:
# NAME              STATUS        PORTS
# vllm-smollm2      Up 2 minutes  0.0.0.0:8009->8000/tcp
# vllm-chatbot      Up 1 minute   0.0.0.0:7860->7860/tcp
```

#### Step 4: Access the Interface

```
Open your browser to: http://localhost:7860

Interface components:
├── Chat display area
├── Message input box
├── Parameter sliders
│   ├── Temperature (0.0 - 2.0)
│   └── Max tokens (50 - 500)
└── Submit button
```

#### Step 5: Test Conversations

Try these conversation flows:

**1. Technical Q&A**
```
You: What is Docker?
Bot: [Response about Docker containers...]

You: How does it differ from virtual machines?
Bot: [Comparative explanation...]
```

**2. Code Assistance**
```
You: Write a Python function to reverse a string
Bot: [Code example...]

You: Can you add error handling to it?
Bot: [Enhanced code...]
```

**3. Creative Tasks**
```
You: Write a short story about AI
Bot: [Creative story...]
```

### Understanding the Code

Key components of `chatbot.py`:

```python
# 1. Environment configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8009/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceTB/SmolLM2-360M-Instruct")

# 2. OpenAI client initialization
client = OpenAI(
    base_url=VLLM_BASE_URL,  # Points to vllm-cpu container
    api_key="dummy"
)

# 3. Chat function with streaming
def chat(message, history, temperature, max_tokens):
    messages = build_message_history(history, message)

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )

    # Stream response token by token
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
            yield response

# 4. Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    additional_inputs=[
        gr.Slider(0, 2, value=0.7, label="Temperature"),
        gr.Slider(50, 500, value=200, label="Max Tokens")
    ]
)
```

### Understanding the Docker Setup

The chatbot is defined in `docker-compose.yml`:

```yaml
chatbot:
  build:
    context: .
    dockerfile: Dockerfile.chatbot  # Simple Python + Gradio image
  ports:
    - "7860:7860"  # Expose Gradio UI
  environment:
    - VLLM_BASE_URL=http://vllm-cpu:8000/v1  # Container-to-container
    - MODEL_NAME=${MODEL_NAME}
  depends_on:
    - vllm-cpu  # Start after vLLM
  networks:
    - vllm-network  # Same network as vLLM
```

**Key points:**
- Uses service name `vllm-cpu:8000` (not `localhost:8009`)
- Containers communicate on internal network
- External access via `localhost:7860`

### Exercise 8: Customize the Chatbot (15 mins)

Modify `chatbot.py` and rebuild the container:

1. **Add a system prompt**
```python
# Edit chatbot.py
system_message = {
    "role": "system",
    "content": "You are a helpful Docker and vLLM expert."
}
```

2. **Rebuild and restart**
```bash
# Stop the chatbot
docker compose stop chatbot

# Rebuild with changes
docker compose build chatbot

# Start the updated chatbot
docker compose up -d chatbot

# View logs to verify
docker compose logs -f chatbot
```

3. **Test your changes**
Open http://localhost:7860 and verify the new behavior

---

## Part 6: Performance Tuning and Optimization (20 mins)

### Monitoring Performance

#### Real-time Metrics

```bash
# Container resource usage
docker stats vllm-smollm2

# Output:
# CONTAINER ID   NAME            CPU %    MEM USAGE / LIMIT
# abc123         vllm-smollm2    45.2%    4.2GB / 8GB
```

#### API Metrics

```bash
# Run the test suite
python test_vllm.py

# Expected output:
# ✓ Health check passed
# ✓ Completion succeeded (2.34s)
# ✓ Chat completion succeeded (3.12s)
# ✓ Streaming succeeded (2.87s)
#
# Performance Summary:
#   Average: 2.78s
#   Min: 2.34s
#   Max: 3.12s
```

### Optimization Experiments

#### Experiment 1: Thread Count

```bash
# Stop the service
docker compose down

# Edit .env
OMP_THREADS=4  # Increase from 2

# Restart and test
docker compose up -d
python test_vllm.py

# Compare performance
```

**Expected Results:**
- Higher CPU usage
- Possibly faster inference (on multi-core systems)
- Watch for thermal throttling

#### Experiment 2: Model Size

```bash
# Try smaller model
MODEL_NAME=HuggingFaceTB/SmolLM2-135M-Instruct
MEMORY_LIMIT=4G

# Or larger model
MODEL_NAME=HuggingFaceTB/SmolLM2-1.7B-Instruct
MEMORY_LIMIT=12G

# Restart and compare
docker compose down && docker compose up -d
```

#### Experiment 3: Context Length

```bash
# Shorter context = less memory
MAX_MODEL_LEN=1024

# Longer context = more memory
MAX_MODEL_LEN=4096

# Test impact on memory usage
```

### Exercise 9: Find Your Optimal Settings (10 mins)

Use this worksheet to document your experiments:

```
System Specs:
- CPU: ___________
- RAM: ___________
- OS: ___________

Experiment Results:
┌──────────────┬─────────┬────────────┬──────────────┐
│ OMP_THREADS  │ CPU %   │ Memory MB  │ Latency (s)  │
├──────────────┼─────────┼────────────┼──────────────┤
│ 2 (default)  │         │            │              │
│ 4            │         │            │              │
│ 6            │         │            │              │
└──────────────┴─────────┴────────────┴──────────────┘

Best configuration for your system:
OMP_THREADS=___
CPU_LIMIT=___
MEMORY_LIMIT=___
```

---

## Part 7: Production Considerations (15 mins)

### Deployment Checklist

- [ ] Resource limits configured
- [ ] Health checks enabled
- [ ] Logging configured
- [ ] Monitoring setup
- [ ] Backup/recovery plan
- [ ] Security hardening
- [ ] API authentication (if needed)
- [ ] Rate limiting
- [ ] Model versioning

### Security Best Practices

```yaml
# 1. Run as non-root user
user: "1000:1000"

# 2. Read-only root filesystem
read_only: true

# 3. Drop capabilities
cap_drop:
  - ALL

# 4. Bind to localhost only (for local dev)
ports:
  - "127.0.0.1:8009:8000"

# 5. Use secrets for sensitive data
secrets:
  - hf_token
```

### Monitoring and Logging

```bash
# Centralized logging
docker compose logs vllm-cpu > vllm.log

# Log rotation
docker compose up -d --log-opt max-size=10m --log-opt max-file=3

# Export metrics
curl http://localhost:8009/metrics
```

### Scaling Strategies

**Horizontal Scaling:**
```yaml
# docker-compose.yml
services:
  vllm-1:
    # Model A on port 8009
  vllm-2:
    # Model B on port 8010
  nginx:
    # Load balancer
```

**Vertical Scaling:**
```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'      # More CPU
      memory: 16G      # More RAM
```

---

## Workshop Summary

### What We Covered

1. ✓ Compared default vs. optimized vLLM images
2. ✓ Understood Dockerfile optimization techniques
3. ✓ Built custom images with Docker Compose
4. ✓ Deployed vLLM with OpenAI-compatible API
5. ✓ Created a Gradio chatbot interface
6. ✓ Performed optimization experiments
7. ✓ Discussed production considerations

### Key Takeaways

**Technical:**
- NUMA patch is critical for containerized vLLM
- Thread tuning significantly impacts performance
- Resource limits prevent system overload
- OpenAI API compatibility enables easy integration

**Best Practices:**
- Always test with your actual workload
- Monitor resource usage continuously
- Start with conservative settings
- Document your configuration

**Performance:**
- Smaller models = faster inference
- Thread count depends on CPU architecture
- Memory limits prevent OOM crashes
- Streaming improves perceived responsiveness

### Next Steps

**Immediate:**
1. Experiment with different models
2. Try the chatbot with your use cases
3. Optimize for your hardware
4. Integrate with your applications

**Advanced:**
- Multi-model serving
- Fine-tune models for your domain
- Implement caching layers
- Add authentication/authorization
- Deploy to production environments

**Resources:**
- vLLM documentation: https://docs.vllm.ai/
- HuggingFace models: https://huggingface.co/HuggingFaceTB
- Gradio docs: https://gradio.app/
- Docker best practices: https://docs.docker.com/

---

## Troubleshooting Guide

### Common Issues

**Issue 1: Container won't start**
```bash
# Check logs
docker compose logs vllm-cpu

# Common causes:
# - Insufficient memory (increase MEMORY_LIMIT)
# - Port conflict (change VLLM_PORT)
# - Model download failed (check internet)
```

**Issue 2: Slow performance**
```bash
# Check CPU usage
docker stats vllm-smollm2

# Solutions:
# - Increase OMP_THREADS
# - Use smaller model
# - Reduce concurrent requests (MAX_NUM_SEQS)
```

**Issue 3: Out of memory**
```bash
# Solutions:
# 1. Switch to smaller model
MODEL_NAME=HuggingFaceTB/SmolLM2-135M-Instruct

# 2. Reduce context length
MAX_MODEL_LEN=1024

# 3. Increase Docker memory limit
MEMORY_LIMIT=12G
```

**Issue 4: Gradio connection error**
```bash
# Check vLLM is running
curl http://localhost:8009/health

# Verify port in chatbot.py matches .env
# base_url="http://localhost:8009/v1"
```

---

## Additional Exercises

### Challenge 1: Multi-Model Setup
Deploy two different models simultaneously on different ports.

### Challenge 2: Custom System Prompt
Create a specialized chatbot (e.g., code reviewer, technical writer).

### Challenge 3: Performance Dashboard
Build a Gradio dashboard showing real-time metrics.

### Challenge 4: API Gateway
Implement rate limiting and authentication.

### Challenge 5: Model Comparison
Create a UI to compare responses from different models.

---

## Feedback and Questions

**Discussion Topics:**
1. What surprised you about vLLM performance?
2. How would you use this in your projects?
3. What optimizations worked best for your system?
4. What additional features would you add?

**Share Your Results:**
- Optimal configuration for your hardware
- Interesting chatbot conversations
- Performance benchmarks
- Creative use cases

---

## Appendix: Reference Commands

```bash
# Build and start
docker compose build
docker compose up -d

# Monitor
docker compose logs -f vllm-cpu
docker stats vllm-smollm2

# Test
curl http://localhost:8009/health
python test_vllm.py
python chatbot.py

# Stop and clean
docker compose down
docker compose down -v  # Remove volumes too

# Restart with new config
docker compose down
# Edit .env
docker compose up -d
```

## License and Attribution

This workshop material is provided as-is for educational purposes.
- vLLM: Apache 2.0 License
- SmolLM2: Apache 2.0 License
- Gradio: Apache 2.0 License
