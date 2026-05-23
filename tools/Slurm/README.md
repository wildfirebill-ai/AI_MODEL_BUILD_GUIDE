# Slurm — HPC Workload Manager for GPU Clusters

Slurm is a highly scalable workload manager for Linux clusters. It handles job scheduling, resource allocation, and parallel task execution across GPU-equipped nodes, making it the de facto standard for HPC and AI training infrastructure.

## Key Concepts

- **Partition** — a named set of nodes (gpu, cpu, highmem)
- **Job** — a resource allocation request
- **Job Array** — a collection of related jobs indexed by task ID
- **QoS** — Quality of Service (priority, limits, preemption)

## Common Commands

```bash
sbatch --partition=gpu --qos=high --gpus=4 --cpus-per-task=16 train.sh
srun --partition=gpu --gpus=2 --cpus-per-task=8 --time=01:00:00 --pty bash
squeue -u $USER
sacct -j 123456 --format=JobID,JobName,State,ExitCode,Elapsed,ReqGPUS,AllocGPUS
scancel 123456
```

## Job Arrays for Hyperparameter Sweeps

```bash
#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-15

PARAMS=(0.001 0.005 0.01 0.05 0.1 0.5 1.0)
LR=${PARAMS[$SLURM_ARRAY_TASK_ID]}
python train.py --lr $LR --run_id "sweep_${SLURM_ARRAY_TASK_ID}"
```

## Multi-Node Distributed Training

```bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

srun python -m torch.distributed.run \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_distributed.py --epochs 100 --batch_size 256
```

## Integration: Submit from Python

```python
import subprocess

def submit_job(name: str, script: str, gpus: int = 1, partition: str = "gpu") -> int:
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --gpus={gpus}
#SBATCH --cpus-per-task={gpus * 8}
#SBATCH --time=04:00:00
#SBATCH --output=logs/{name}_%j.out
cd {subprocess.check_output(["pwd"]).decode().strip()}
python {script}
"""
    path = f"{name}.sbatch"
    with open(path, "w") as f:
        f.write(sbatch_content)
    result = subprocess.run(["sbatch", path], capture_output=True, text=True)
    job_id = int(result.stdout.strip().split()[-1])
    print(f"Submitted job {job_id}: {name}")
    return job_id

def monitor_job(job_id: int):
    import time
    while True:
        r = subprocess.run(["sacct", "-j", str(job_id), "--format=State",
                            "--noheader", "-P"], capture_output=True, text=True)
        state = r.stdout.strip().split("\n")[0] if r.stdout.strip() else "?"
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            return state
        time.sleep(30)

if __name__ == "__main__":
    jid = submit_job("train_vit", "train.py --config vit_base.yaml", gpus=4)
    print(f"Final status: {monitor_job(jid)}")
```

## Best Practices

- Use job arrays for embarrassingly parallel hyperparameter sweeps.
- Request 8-16 CPUs per GPU for optimal data-loading throughput.
- Set realistic walltimes — long walltimes reduce scheduler flexibility.
- Use `sacct` for post-mortem diagnostics rather than output logs.
