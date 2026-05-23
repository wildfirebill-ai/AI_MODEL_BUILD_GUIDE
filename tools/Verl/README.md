# VeRL (Voltron-Enhanced Reinforcement Learning)

**Purpose:** RLHF training framework with GRPO (Group Relative Policy Optimization) for reasoning model training (used by DeepSeek-R1).

## Installation

```powershell
pip install verl
```

## Usage

```python
from verl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_model=reward_model,
    train_dataset=reasoning_dataset,
    learning_rate=1e-6,
    beta=0.04,          # KL penalty coefficient
    group_size=8,       # Samples per prompt for advantage estimation
    max_length=32768,   # Long reasoning traces
    temperature=0.8,
)

trainer.train()
```

## GRPO vs PPO

| Feature | PPO | GRPO |
|---------|-----|------|
| Value model | Required | Not needed |
| Group sampling | No | Yes (group-based advantage) |
| KL penalty | Adaptive | Fixed beta |
| Memory usage | Higher (value model) | Lower |
| Used by | OpenAI, Anthropic | DeepSeek-R1 |

## Key Features

- **Distributed** training across multiple GPUs/nodes
- **Hybrid engine** (training + inference for generation)
- **Long-context** support (32K+)
- **Flash Attention** integrated

## Documentation

- https://github.com/volcengine/verl
