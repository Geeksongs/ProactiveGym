"""
Debug test script for ProactiveGym.
Automatically runs a few steps to check if the environment works correctly.
"""

import json
from datetime import datetime
from proactivegym import ProactiveEnv, get_demo_config

def main():
    print("=" * 70)
    print("ProactiveGym Debug Test")
    print("=" * 70)

    # Create config
    config = get_demo_config()
    config.verbose = True

    print(f"\nConfig:")
    print(f"  Model: {config.model_name}")
    print(f"  Reward Model: {config.reward_model_name}")
    print(f"  Max Steps: {config.max_steps}")
    print(f"  API Key: {config.api_key[:20]}...")


    #这里是openai的gym里面的最重要的逻辑，现在开始
    # Create environment
    print("\n[1] Creating environment...")
    env = ProactiveEnv(config)

    # Reset with coding theme
    print("\n[2] Resetting environment (theme: coding)...")
    obs, info = env.reset(options={"theme": "coding"})

    print("\n" + "-" * 70)
    print("Initial State:")
    print("-" * 70)
    print(f"Scenario: {info.get('scenario_title', 'N/A')}")
    print(f"User Goal: {info.get('user_goal', 'N/A')}")
    print(f"\nInitial Events:\n{obs['events']}")
    print("-" * 70)

    # 保存初始状态
    trajectory_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_name": config.model_name,
            "reward_model_name": config.reward_model_name,
            "max_steps": config.max_steps,
        },
        "user_profile": info.get("user_profile", {}),
        "scenario": {
            "title": info.get("scenario_title", ""),
            "user_goal": info.get("user_goal", ""),
            "theme": "coding",
        },
        "initial_observation": {
            "events": obs["events"],
            "scenario": obs["scenario"],
            "goal": obs["goal"],
        },
        "steps": [],
    }

    # Test actions
    test_actions = [
        "[silent]",
        "[predict] Help user set up the development environment",
        "[silent]",
    ]

    for i, action in enumerate(test_actions):
        print(f"\n[Step {i+1}] Taking action: {action}")
        print("-" * 40)

        obs, reward, terminated, truncated, info = env.step(action)

        step_data = {
            "step": i + 1,
            "action": action,
            "reward": reward,
            "observation": {
                "events": obs["events"],
                "feedback": obs["feedback"],
                "step_count": obs["step_count"],
            },
            "terminated": terminated,
            "truncated": truncated,
        }
        trajectory_data["steps"].append(step_data)

        print(f"Reward: {reward:.4f}")
        print(f"Feedback: {obs['feedback']}")
        print(f"New Events:\n{obs['events']}")

        if terminated or truncated:
            print("\n*** Episode terminated early ***")
            break

    # Get metrics
    metrics = env.get_metrics()
    trajectory_data["metrics"] = metrics
    trajectory_data["judgment_history"] = env.judgment_history
    trajectory_data["action_history"] = env.action_history

    # Print summary
    print("\n" + "=" * 70)
    print("Trajectory Summary")
    print("=" * 70)

    total_reward = sum(s["reward"] for s in trajectory_data["steps"])
    for step in trajectory_data["steps"]:
        print(f"\nStep {step['step']}:")
        print(f"  Action: {step['action']}")
        print(f"  Reward: {step['reward']:.4f}")
        print(f"  Feedback: {step['observation']['feedback']}")

    print(f"\nTotal Reward: {total_reward:.4f}")

    print("\n" + "=" * 70)
    print("Episode Metrics")
    print("=" * 70)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 保存轨迹到 data 文件夹
    filename = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file = f"data/{filename}"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(trajectory_data, f, ensure_ascii=False, indent=2)

    print(f"\n轨迹已保存到: {output_file}")

    env.close()
    print("Debug test completed!")

if __name__ == "__main__":
    main()
