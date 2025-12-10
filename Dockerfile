FROM openeuler/vllm-cpu:0.9.1-oe2403lts

# Patch the cpu_worker.py to handle zero NUMA nodes
RUN sed -i 's/cpu_count_per_numa = cpu_count \/\/ numa_size/cpu_count_per_numa = cpu_count \/\/ numa_size if numa_size > 0 else cpu_count/g' \
    /workspace/vllm/vllm/worker/cpu_worker.py

ENV VLLM_TARGET_DEVICE=cpu \
    VLLM_CPU_KVCACHE_SPACE=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
