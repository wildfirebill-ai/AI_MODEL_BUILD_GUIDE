# Habitat — Meta's Embodied AI Simulation Platform

[Habitat](https://aihabitat.org) is a high-performance 3D simulator for embodied AI research, including navigation, rearrangement, and manipulation tasks.

## Installation

```bash
# Habitat-Sim (3D simulator)
pip install habitat-sim

# Habitat-Lab (training framework)
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e .

# Download test scenes
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path ./data
```

## Quick Start — Navigate in a Scene

```python
import habitat_sim
import numpy as np

# Create a simulator
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.gpu_device_id = 0
sim_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"

agent_cfg = habitat_sim.agent.AgentConfiguration()
cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

sim = habitat_sim.Simulator(cfg)
agent = sim.initialize_agent(0)

# Move forward
obs = agent.act(habitat_sim.ActionSpec("move_forward", 0.25))
print(f"RGB shape: {obs['color'].shape}")  # (480, 640, 4)

# Rotate
obs = agent.act(habitat_sim.ActionSpec("turn_left", 30.0))

# Get position
print(agent.get_state().position)

sim.close()
```

## Habitat-Lab — Task-Based Training

```python
# train.py
import habitat
from habitat.config.default import get_config

cfg = habitat.get_config("configs/tasks/pointnav.yaml")
env = habitat.Env(config=cfg)

observations = env.reset()
print(observations.keys())  # ['rgb', 'depth', 'gps', 'compass', 'pointgoal']

done = False
while not done:
    action = env.action_space.sample()
    observations, reward, done, info = env.step(action)

env.close()
```

## Navigation Tasks (PointNav, ObjectNav)

```python
import habitat
from habitat.config.default import get_config

# PointNav: navigate to a target GPS coordinate
cfg = habitat.get_config("configs/tasks/pointnav.yaml")

# ObjectNav: navigate to a target object category
cfg = habitat.get_config("configs/tasks/objectnav.yaml")

env = habitat.Env(config=cfg)
obs = env.reset()
```

## Sensor Configuration

```python
from habitat.config.default import get_config

cfg = habitat.get_config("configs/tasks/pointnav.yaml")
cfg.defrost()

# Configure RGB-D sensors
cfg.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
cfg.RGB_SENSOR.WIDTH = 256
cfg.RGB_SENSOR.HEIGHT = 256
cfg.DEPTH_SENSOR.WIDTH = 256
cfg.DEPTH_SENSOR.HEIGHT = 256

# Add GPS+Compass
cfg.SENSORS += ["GPS_SENSOR", "COMPASS_SENSOR"]
cfg.freeze()
```

## Rearrangement Tasks

```python
import habitat
from habitat.config.default import get_config

# Rearrangement: pick-and-place objects
cfg = habitat.get_config("configs/tasks/rearrangement/rearrangement_easy.yaml")
env = habitat.Env(config=cfg)

obs = env.reset()
# Observations include: rgb, depth, object_bbox, start_receptacle, target_receptacle
```

## Multi-Agent Environments

```python
from habitat import MultiAgentEnv
from habitat.config.default import get_config

cfg = habitat.get_config("configs/tasks/pointnav_multiagent.yaml")
env = MultiAgentEnv(config=cfg)

observations = env.reset()
# Each agent has its own observations
for i, obs in enumerate(observations):
    print(f"Agent {i} position: {obs['gps']}")
```

## Scene Datasets

| Dataset        | Description                    | Download                    |
|----------------|--------------------------------|-----------------------------|
| HM3D           | 1000+ real-world scans        | `--uids hm3d`              |
| Gibson         | 572 real-world scans          | `--uids gibson`            |
| Matterport3D   | 90 building-scale scans       | Requires license           |
| Replica        | 18 photorealistic apartments  | `--uids replica_cad`       |
| Habitat-Matterport| Replica of 10M+ frames    | `--uids habitat_matterport`|

```bash
python -m habitat_sim.utils.datasets_download --uids hm3d --data-path ./data
```

## Render Modes

```python
# Batch rendering for training
obs = sim.get_sensor_observations()
rgb = obs["color_sensor"]     # (H, W, 4) RGBA
depth = obs["depth_sensor"]   # (H, W, 1) meters
semantic = obs["semantic_sensor"]  # (H, W, 1) category IDs
```

## Use Cases

- **Robotics simulation** — train navigation and manipulation policies
- **Embodied AI research** — benchmark agents on perception and planning
- **Multi-agent environments** — collaborative or competitive embodied tasks
- **Rearrangement** — train robots to organize scenes
- **Sim-to-real transfer** — train in Habitat, deploy on real robots
