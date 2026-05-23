# Docker

**Purpose:** Containerization for reproducible AI/ML environments and deployment.

## Installation

- Download from https://www.docker.com/products/docker-desktop/

## Usage

### Basic Container for Training

```dockerfile
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "train.py"]
```

### Build and Run

```powershell
docker build -t wfb-trainer .
docker run --gpus all -it --rm wfb-trainer
```

### Docker Compose for Multi-Service

```yaml
version: "3.8"
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - MODEL_PATH=/model
      - TENSOR_PARALLEL_SIZE=4
    volumes:
      - ./model:/model
    ports:
      - "8000:8000"

  redis:
    image: redis:alpine

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### NVIDIA Container Toolkit

Required for GPU access inside containers:
```powershell
# Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Documentation

- https://docs.docker.com/
