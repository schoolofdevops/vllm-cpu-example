# Workshop Setup Checklist

Complete these steps **before** the workshop to ensure a smooth experience.

## Prerequisites

### System Requirements

- [ ] **Operating System**: macOS 10.15+ (Catalina or later)
- [ ] **RAM**: Minimum 8GB, 16GB recommended
- [ ] **Disk Space**: At least 15GB free
- [ ] **CPU**: Multi-core processor (4+ cores recommended)
- [ ] **Internet**: Stable connection for downloading models (~2GB)

### Software Requirements

#### 1. Docker Desktop

- [ ] Install Docker Desktop for Mac
  ```bash
  # Download from: https://www.docker.com/products/docker-desktop
  # Or install via Homebrew:
  brew install --cask docker
  ```

- [ ] Verify Docker installation:
  ```bash
  docker --version
  # Expected: Docker version 20.10.0 or later

  docker compose version
  # Expected: Docker Compose version v2.0.0 or later
  ```

- [ ] Configure Docker Desktop resources:
  - Open Docker Desktop → Settings → Resources
  - **Memory**: Set to at least 8GB
  - **CPUs**: Set to at least 4 cores
  - **Disk**: Ensure at least 15GB available

- [ ] Test Docker:
  ```bash
  docker run hello-world
  # Should download and run successfully
  ```

#### 2. Development Tools

- [ ] Install curl (usually pre-installed on macOS):
  ```bash
  curl --version
  ```

- [ ] Install jq for JSON formatting (optional but helpful):
  ```bash
  brew install jq
  # Or download from: https://stedolan.github.io/jq/
  ```

- [ ] Install a text editor or IDE:
  - [ ] VS Code (recommended)
  - [ ] Sublime Text
  - [ ] Vim/Nano
  - [ ] Any editor you're comfortable with

## Pre-Workshop Tasks

### 1. Clone/Download Workshop Files

- [ ] Navigate to workshop directory:
  ```bash
  cd /path/to/vllm-cpu
  ```

- [ ] Verify all required files are present:
  ```bash
  ls -la
  # Should see:
  # - Dockerfile
  # - docker-compose.yml
  # - .env
  # - chatbot.py
  # - test_vllm.py
  # - WORKSHOP.md
  # - requirements-workshop.txt
  ```

### 2. Review Configuration

- [ ] Check `.env` file:
  ```bash
  cat .env
  # Verify VLLM_PORT=8009
  # Note the MODEL_NAME setting
  ```

- [ ] Understand the settings:
  - Default model: SmolLM2-360M-Instruct
  - Default port: 8009
  - Default memory limit: 8G

### 3. Pre-download Docker Base Image (Optional)

This step is optional but saves time during the workshop:

- [ ] Pull the base image:
  ```bash
  docker pull openeuler/vllm-cpu:0.9.1-oe2403lts
  ```

- [ ] Verify:
  ```bash
  docker images | grep openeuler
  ```

### 4. Build the Image (Optional)

You can build ahead of time to save workshop time:

- [ ] Build the vLLM image:
  ```bash
  docker compose build
  ```

- [ ] Verify the build:
  ```bash
  docker images | grep vllm-cpu-optimized
  # Should show: vllm-cpu-optimized:latest
  ```

**Note**: If you skip this step, the image will be built automatically during the workshop when you run `docker compose up`.

### 5. Pre-download Model (Optional, Saves 5-10 Minutes)

This step downloads the model ahead of time:

- [ ] Start the container once to download the model:
  ```bash
  docker compose up -d
  ```

- [ ] Monitor the download:
  ```bash
  docker compose logs -f vllm-cpu
  # Wait for: "Downloading model..." and completion
  # This may take 5-10 minutes
  ```

- [ ] Wait for service to be ready:
  ```bash
  curl http://localhost:8009/health
  # Wait until you see: {"status":"ok"}
  ```

- [ ] Stop the service:
  ```bash
  docker compose down
  ```

**Note**: The model (~1.3GB) is now cached in a Docker volume and won't need to be downloaded again during the workshop.

## Verification Checklist

Run these commands to verify everything is working:

### Docker Verification

```bash
# 1. Docker is running
docker info
# Should show: Server Version, Storage Driver, etc.

# 2. Docker Compose works
docker compose version
# Should show: version 2.x.x or later

# 3. Can pull images
docker pull hello-world
docker run hello-world
# Should print "Hello from Docker!"
```

### System Resources

```bash
# Check available disk space
df -h | grep -E "Filesystem|/$"
# Ensure at least 15GB free

# Check available memory
# Open Activity Monitor (macOS) and verify:
# - At least 8GB RAM available
# - CPU not heavily loaded
```

## Quick Start Test

If you completed the optional pre-download steps, verify everything works:

```bash
# 1. Start vLLM
docker compose up -d

# 2. Wait for health check
curl http://localhost:8009/health
# Should return: {"status":"ok"}

# 3. Test API
curl http://localhost:8009/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "prompt": "Hello",
    "max_tokens": 10
  }'
# Should return JSON with generated text

# 4. Stop service
docker compose down
```

## Troubleshooting

### Docker Issues

**Problem**: Docker Desktop won't start
- **Solution**: Restart your computer, check system requirements

**Problem**: "Cannot connect to Docker daemon"
- **Solution**: Start Docker Desktop application

**Problem**: "No space left on device"
- **Solution**: Free up disk space, clean Docker:
  ```bash
  docker system prune -a
  ```

### Python Issues

**Problem**: "No module named 'gradio'"
- **Solution**: Install requirements:
  ```bash
  pip install -r requirements-workshop.txt
  ```

**Problem**: "pip: command not found"
- **Solution**: Use `pip3` instead of `pip`

### Port Conflicts

**Problem**: "Port 8009 already in use"
- **Solution**:
  1. Find what's using the port: `lsof -i :8009`
  2. Stop that service or change VLLM_PORT in `.env`

## Workshop Day Checklist

On the day of the workshop:

- [ ] Laptop fully charged or connected to power
- [ ] Docker Desktop is running
- [ ] All required files are in the workshop directory
- [ ] Python environment is activated (if using venv)
- [ ] Internet connection is stable
- [ ] Text editor is ready
- [ ] Terminal/command line is accessible

## Getting Help

If you encounter issues during setup:

1. Check the Troubleshooting section above
2. Review the error messages carefully
3. Google the specific error message
4. Bring questions to the workshop

## Resources

- Docker Desktop: https://www.docker.com/products/docker-desktop
- Python: https://www.python.org/downloads/
- Gradio: https://gradio.app/
- vLLM: https://docs.vllm.ai/
- OpenAI API: https://platform.openai.com/docs/api-reference

## Estimated Setup Time

- Basic setup (Docker + Python): 15-20 minutes
- With optional pre-download: 30-40 minutes
- Full verification: 5-10 minutes

**Total**: 20-50 minutes depending on options chosen

---

**Ready for the workshop?** Make sure all items in the "Prerequisites" and "Verification Checklist" sections are complete!
