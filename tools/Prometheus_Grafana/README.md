# Prometheus + Grafana

**Purpose:** Metrics collection and visualization for monitoring LLM inference servers.

---

## Prometheus

### Installation

```powershell
# Windows: download from https://prometheus.io/download/
# Or use Docker:
docker run -d -p 9090:9090 prom/prometheus
```

### Config (prometheus.yml)

```yaml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
```

---

## Grafana

### Installation

```powershell
docker run -d -p 3000:3000 grafana/grafana
# Default: admin/admin
```

### Key Dashboards

- **LLM Inference**: latency, throughput, tokens/sec, GPU utilization
- **Training**: loss curves, LR schedule, GPU memory, throughput
- **Hardware**: GPU temperature, power, memory bandwidth

---

## Python Client

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

REQUESTS = Counter("llm_requests_total", "Total requests")
LATENCY = Histogram("llm_latency_seconds", "Latency", buckets=[0.1, 0.5, 1.0, 2.0, 5.0])
MEMORY = Gauge("llm_gpu_memory_gb", "GPU memory used")

start_http_server(8000)
```
