#!/usr/bin/env python3
"""
Demo script for OpenSandbox integration with ProactiveGym.

This script demonstrates how to:
1. Create a desktop sandbox
2. Capture screenshots (observations)
3. Execute mouse/keyboard actions
4. Use for RL training workflows
"""

import asyncio
from pathlib import Path

from desktop_sandbox import DesktopSandbox, create_parallel_sandboxes
from config import SandboxConfig


async def basic_demo():
    """Basic demonstration of sandbox capabilities."""
    print("=== OpenSandbox Basic Demo ===\n")

    # Create sandbox with default config
    config = SandboxConfig()
    sandbox = DesktopSandbox(config)

    print("Creating sandbox...")
    sandbox_id = await sandbox.create()
    print(f"Sandbox created: {sandbox_id}")

    try:
        # Step 1: Get observation (screenshot)
        print("\n[Step 1] Capturing screenshot...")
        screenshot = await sandbox.screenshot()
        Path("/tmp/demo_obs1.png").write_bytes(screenshot)
        print(f"Screenshot saved: {len(screenshot)} bytes")

        # Step 2: Execute action (click on Applications menu)
        print("\n[Step 2] Clicking on Applications menu (50, 12)...")
        await sandbox.click(50, 12)
        await asyncio.sleep(1)

        # Step 3: Get next observation
        print("\n[Step 3] Capturing next screenshot...")
        screenshot = await sandbox.screenshot()
        Path("/tmp/demo_obs2.png").write_bytes(screenshot)
        print(f"Screenshot saved: {len(screenshot)} bytes")

        # Step 4: Type some text
        print("\n[Step 4] Pressing Escape and opening terminal...")
        await sandbox.press_key("escape")
        await asyncio.sleep(0.5)

        # Open terminal with keyboard shortcut
        await sandbox.hotkey("ctrl", "alt", "t")
        await asyncio.sleep(2)

        # Type a command
        print("\n[Step 5] Typing command in terminal...")
        await sandbox.type_text("echo 'Hello from ProactiveGym!'")
        await sandbox.press_key("enter")
        await asyncio.sleep(1)

        # Final screenshot
        print("\n[Step 6] Final screenshot...")
        screenshot = await sandbox.screenshot()
        Path("/tmp/demo_final.png").write_bytes(screenshot)
        print(f"Screenshot saved: {len(screenshot)} bytes")

        print("\n=== Demo Complete ===")
        print("Screenshots saved to /tmp/demo_*.png")

    finally:
        print("\nCleaning up...")
        await sandbox.close()
        print("Sandbox closed.")


async def rl_step_demo():
    """Demonstrate a single RL step: observe -> act -> observe."""
    print("=== RL Step Demo ===\n")

    sandbox = DesktopSandbox()
    await sandbox.create()

    try:
        # Observation 1
        obs1 = await sandbox.screenshot()
        print(f"Observation 1: {len(obs1)} bytes")

        # Action: click somewhere
        await sandbox.click(640, 400)  # Center of screen
        await asyncio.sleep(0.5)

        # Observation 2
        obs2 = await sandbox.screenshot()
        print(f"Observation 2: {len(obs2)} bytes")

        # In RL, you would:
        # 1. Convert obs to numpy array / tensor
        # 2. Feed to policy network
        # 3. Get action from policy
        # 4. Execute action in sandbox
        # 5. Get reward from reward model
        # 6. Repeat

        print("\nRL integration pattern:")
        print("  obs = await sandbox.screenshot()")
        print("  action = policy(obs)")
        print("  await sandbox.click(action.x, action.y)")
        print("  next_obs = await sandbox.screenshot()")
        print("  reward = reward_model(obs, action, next_obs)")

    finally:
        await sandbox.close()


async def parallel_demo():
    """Demonstrate parallel sandbox creation for distributed training."""
    print("=== Parallel Sandboxes Demo ===\n")

    num_sandboxes = 2  # Adjust based on available resources

    print(f"Creating {num_sandboxes} sandboxes in parallel...")
    sandboxes = await create_parallel_sandboxes(num_sandboxes)

    print(f"Created {len(sandboxes)} sandboxes:")
    for i, sb in enumerate(sandboxes):
        print(f"  Sandbox {i+1}: {sb.sandbox_id}")

    try:
        # Take screenshots from all sandboxes concurrently
        print("\nCapturing screenshots from all sandboxes...")
        screenshots = await asyncio.gather(
            *[sb.screenshot() for sb in sandboxes]
        )

        for i, screenshot in enumerate(screenshots):
            Path(f"/tmp/parallel_obs_{i+1}.png").write_bytes(screenshot)
            print(f"  Sandbox {i+1}: {len(screenshot)} bytes")

        print("\nParallel sandboxes ready for distributed RL training!")

    finally:
        print("\nCleaning up...")
        await asyncio.gather(*[sb.close() for sb in sandboxes])
        print("All sandboxes closed.")


if __name__ == "__main__":
    import sys

    demos = {
        "basic": basic_demo,
        "rl": rl_step_demo,
        "parallel": parallel_demo,
    }

    if len(sys.argv) > 1 and sys.argv[1] in demos:
        asyncio.run(demos[sys.argv[1]]())
    else:
        print("Usage: python demo.py [basic|rl|parallel]")
        print("\nRunning basic demo...")
        asyncio.run(basic_demo())
