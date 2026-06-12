<p align="center">
  <img src="image.png" width="175" alt="ProactiveGym Logo">
</p>

# ProactiveGym

A Gymnasium environment for training proactive agents that anticipate user needs and offer timely assistance.

## Overview

ProactiveGym enables training LLM agents to:
1. **Observe** user activities and environment events
2. **Decide** when to proactively intervene
3. **Learn** from accept/reject feedback via reinforcement learning

Inspired by:
- [ProactiveAgent](https://github.com/thunlp/ProactiveAgent) (ICLR 2025)
- [UserRL](https://github.com/SalesforceAIResearch/UserRL)

## Installation

```bash
# Create conda environment
conda create -n proactivegym python=3.10
conda activate proactivegym

# Install package
cd ProactiveGym
pip install -e .
```

## Quick Start

```python
import proactivegym
from proactivegym import ProactiveEnv, get_default_config

# Set your API key
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Create environment
config = get_default_config()
env = ProactiveEnv(config)

# Run an episode
obs, info = env.reset(options={"theme": "coding"})

while True:
    # Your agent decides action based on observation
    # action = your_agent.predict(obs)

    # Example: stay silent
    action = "[silent]"

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
```

## Action Space

| Action | Format | Description |
|--------|--------|-------------|
| **Predict** | `[predict] <task>` | Proactively suggest a task to help the user |
| **Silent** | `[silent]` | Stay silent, don't intervene |
| **Finish** | `[finish]` | End the episode |

## Reward Structure

| Scenario | Reward | Description |
|----------|--------|-------------|
| Predict + Accept | +1.0 | User accepted your helpful suggestion |
| Predict + Reject | -0.1 | False alarm - user didn't need this help |
| Silent + No Need | +0.001 | Correctly stayed quiet when user was fine |
| Silent + Needed | -0.3 | Missed opportunity to help |

## Observation Space

```python
{
    "events": "Recent environment events (text)",
    "scenario": "Current scenario description",
    "goal": "User's goal",
    "feedback": "Feedback from last action",
    "step_count": int
}
```

## Configuration

```python
from proactivegym import ProactiveGymConfig

config = ProactiveGymConfig(
    # LLM settings
    api_key="your-key",
    model_name="gpt-4o-mini",

    # Environment settings
    max_steps=150,
    max_events_per_step=5,

    # Reward settings
    accept_reward=1.0,
    reject_reward=-0.1,
    correct_silence_reward=0.001,
    missed_opportunity_reward=-0.3,

    # Themes
    themes=["coding", "writing", "daily_life"]
)
```

## Interactive Testing

```bash
export OPENAI_API_KEY="your-key"
python test_human.py
```

## Debug Testing

Run automated tests to verify the environment works correctly and inspect trajectory data:

```bash
# Activate conda environment
conda activate proactivegym

# Run debug test
python debug_test.py
```

This will:
1. Create a ProactiveEnv with demo config
2. Generate a coding scenario via LLM
3. Execute test actions (`[silent]`, `[predict]`, `[silent]`)
4. Display rewards and feedback for each step
5. Save trajectory to `data/trajectory_YYYYMMDD_HHMMSS.json`

Example output:
```
======================================================================
ProactiveGym Debug Test
======================================================================
Config:
  Model: gpt-5.4-nano-2026-03-17
  Max Steps: 10

[1] Creating environment...
[2] Resetting environment (theme: coding)...

Scenario: Refactoring a Node.js Service with Failing Tests
User Goal: Complete the refactor, make all tests pass...

[Step 1] Taking action: [silent]
Reward: -0.0090
Feedback: You correctly stayed silent. The user didn't need interruption.

[Step 2] Taking action: [predict] Help user set up the development environment
Reward: -0.1200
Feedback: User rejected your suggestion.

轨迹已保存到: data/trajectory_20260612_110924.json
```

## Integration with UserRL

This gym follows the UserRL interface and can be integrated for RL training:

```python
# In UserRL's verl/tools/env_manager.py, add:
from proactivegym import ProactiveEnv, get_default_config

def create_proactive_env():
    config = get_default_config()
    return ProactiveEnv(config)
```

## Architecture

```
ProactiveGym/
├── proactivegym/
│   ├── __init__.py
│   ├── config.py              # Configuration
│   └── env/
│       ├── proactive_env.py   # Main Gym environment
│       ├── simulated_user.py  # User simulation + Reward Model
│       ├── env_simulator.py   # Event generation
│       └── prompts.py         # LLM prompts
├── sandbox/                   # OpenSandbox integration
│   ├── __init__.py
│   ├── config.py              # Sandbox configuration
│   ├── desktop_sandbox.py     # Desktop environment wrapper
│   ├── demo.py                # Demo scripts
│   ├── Dockerfile             # Custom desktop image
│   └── README.md              # Sandbox documentation
├── test_human.py              # Interactive testing
└── setup.py
```

## Key Components

### SimulatedUser
- Generates realistic user activities
- Judges agent predictions (accept/reject) using reward model approach
- Determines if user needed help (for evaluating silent actions)

### EnvironmentSimulator
- Generates scenario contexts
- Produces environment events based on user activities
- Maintains environment state

## Research Applications

- **Online RL from Human Feedback**: Each accept/reject provides reward signal
- **Intervention Timing**: Learning when to interrupt vs. stay silent
- **Personalization**: Adapting to different user profiles and working styles
- **Multi-turn Decision Making**: Sequential decisions with long-term consequences

## Citation

If you use this in your research, please cite our work, We would really appreciate it！

```bibtex
@misc{proactivegym2026,
  title={ProactiveGym: A Gymnasium Environment for Training Proactive Agents},
  year={2026},
}
```

## Related Work

- [ProactiveAgent Paper](https://arxiv.org/abs/2410.12361)
- [UserRL Paper](https://arxiv.org/abs/2509.19736)
