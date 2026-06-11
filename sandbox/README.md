# OpenSandbox Integration for ProactiveGym

This module provides desktop sandbox environments for RL training using [OpenSandbox](https://github.com/alibaba/OpenSandbox).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Your Application                     │
│  (Python SDK / .NET SDK / JavaScript SDK / Go SDK)      │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTP API
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  OpenSandbox Server                      │
│                   (localhost:8080)                       │
└─────────────────────────┬───────────────────────────────┘
                          │ Manages
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Docker / Kubernetes Runtime                 │
│                 (Runs containers)                        │
└─────────────────────────────────────────────────────────┘
```

**Key Points:**
- SDK does not call Docker directly - it calls OpenSandbox Server
- OpenSandbox Server manages Docker (configured in `~/.sandbox.toml`)
- Custom Docker images are built once to pre-install software
- SDK references images by name: `Sandbox.create("pythonsonggeek/proactivegym-desktop:latest", ...)`

## Requirements

- OpenSandbox server running locally (default: `localhost:8080`)
- Docker with `pythonsonggeek/proactivegym-desktop:latest` image

## Custom Desktop Image

**Pull from Docker Hub (recommended):**

```bash
docker pull pythonsonggeek/proactivegym-desktop:latest
```

**Or build locally:**

```bash
cd sandbox
docker build -t pythonsonggeek/proactivegym-desktop:latest .
```

**Pre-installed software:**
- Google Chrome (browser, email, calendar, chat)
- LibreOffice (Writer, Calc, Impress)
- Evince (PDF reader)
- VSCode (code editor)
- mss + pyautogui (screenshot and automation tools)

## Quick Start

```python
import asyncio
from sandbox import DesktopSandbox

async def main():
    # Create a desktop sandbox
    sandbox = DesktopSandbox()
    await sandbox.create()

    try:
        # Get observation (screenshot)
        screenshot = await sandbox.screenshot()

        # Execute action
        await sandbox.click(100, 200)

        # Get next observation
        next_screenshot = await sandbox.screenshot()

    finally:
        await sandbox.close()

asyncio.run(main())
```

## API Reference

### DesktopSandbox

Main class for interacting with desktop sandboxes.

#### Methods

| Method | Description |
|--------|-------------|
| `create()` | Create a new sandbox with desktop environment |
| `connect(sandbox_id)` | Connect to an existing sandbox |
| `close()` | Close and cleanup the sandbox |
| `screenshot()` | Capture screenshot (returns PNG bytes) |
| `click(x, y)` | Left-click at coordinates |
| `double_click(x, y)` | Double-click at coordinates |
| `right_click(x, y)` | Right-click at coordinates |
| `type_text(text)` | Type text using keyboard |
| `press_key(key)` | Press a single key |
| `hotkey(*keys)` | Press keyboard shortcut |
| `move_mouse(x, y)` | Move mouse to coordinates |
| `drag(x1, y1, x2, y2)` | Drag from one position to another |
| `scroll(clicks)` | Scroll mouse wheel |
| `run_command(cmd)` | Run shell command |
| `read_file(path)` | Read file from sandbox |
| `write_file(path, content)` | Write file to sandbox |

### SandboxConfig

Configuration options for sandboxes.

```python
from sandbox import SandboxConfig

config = SandboxConfig(
    server_host="localhost",
    server_port=8080,
    cpu="4",
    memory="8Gi",
    screen_width=1280,
    screen_height=800,
)
```

### Parallel Sandboxes

Create multiple sandboxes for distributed training:

```python
from sandbox import create_parallel_sandboxes

sandboxes = await create_parallel_sandboxes(count=4)
```

## Demo

Run the demo scripts:

```bash
# Basic demo
python sandbox/demo.py basic

# RL step demo
python sandbox/demo.py rl

# Parallel sandboxes demo
python sandbox/demo.py parallel
```

## RL Training Integration

```python
async def rl_training_loop(sandbox, policy, reward_model):
    obs = await sandbox.screenshot()

    for step in range(max_steps):
        # Get action from policy
        action = policy.predict(obs)

        # Execute action
        if action.type == "click":
            await sandbox.click(action.x, action.y)
        elif action.type == "type":
            await sandbox.type_text(action.text)

        # Get next observation
        next_obs = await sandbox.screenshot()

        # Calculate reward
        reward = reward_model(obs, action, next_obs)

        # Update policy
        policy.update(obs, action, reward, next_obs)

        obs = next_obs
```
