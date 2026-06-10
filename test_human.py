"""
Human test script for ProactiveGym.

This script allows you to manually interact with the ProactiveGym environment.
"""

import os

# Make sure to set your API key
if not os.getenv("OPENAI_API_KEY"):
    print("Please set OPENAI_API_KEY environment variable")
    print("export OPENAI_API_KEY='your-key-here'")
    exit(1)


if __name__ == "__main__":
    import proactivegym
    from proactivegym import ProactiveEnv, get_demo_config

    # Create config
    config = get_demo_config()

    # Create environment
    env = ProactiveEnv(config)

    # Reset with a specific theme
    print("\nChoose theme:")
    print("1: coding")
    print("2: writing")
    print("3: daily_life")

    theme_choice = input("Enter choice (1-3): ").strip()
    themes = {"1": "coding", "2": "writing", "3": "daily_life"}
    theme = themes.get(theme_choice, "coding")

    obs, info = env.reset(options={"theme": theme})

    print("\n" + "=" * 60)
    print("ProactiveGym Interactive Test")
    print("=" * 60)
    print(f"Scenario: {info['scenario_title']}")
    print(f"User Goal: {info['user_goal']}")
    print("\nInstructions:")
    print("  1: [predict] - Make a prediction (you'll be asked for the task)")
    print("  2: [silent] - Stay silent, don't intervene")
    print("  3: [finish] - End the episode")
    print("=" * 60)

    while True:
        print("\n--- Current Observation ---")
        print(f"Events:\n{obs['events']}")
        print(f"\nFeedback: {obs['feedback']}")
        print(f"Step: {obs['step_count']}")
        print("-" * 40)

        print("\nChoose action:")
        print("  1: [predict] - Suggest a task")
        print("  2: [silent] - Stay silent")
        print("  3: [finish] - End episode")

        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            task = input("Enter your prediction (what help to offer): ")
            action = f"[predict] {task}"
        elif choice == "2":
            action = "[silent]"
        elif choice == "3":
            action = "[finish]"
        else:
            print("Invalid choice, try again")
            continue

        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n>>> Action: {action}")
        print(f">>> Reward: {reward:.3f}")
        print(f">>> Feedback: {obs['feedback']}")

        if terminated or truncated:
            print("\n" + "=" * 60)
            print("Episode Finished!")
            print("=" * 60)

            metrics = env.get_metrics()
            print(f"Total Steps: {metrics['total_steps']}")
            print(f"Total Reward: {metrics['total_reward']:.3f}")
            print(f"Predictions: {metrics['predictions']} (Accepted: {metrics['accepted']}, Rejected: {metrics['rejected']})")
            print(f"Silences: {metrics['silences']} (Correct: {metrics['correct_silence']}, Missed: {metrics['missed_opportunity']})")
            if metrics['predictions'] > 0:
                print(f"Precision: {metrics['precision']:.2%}")

            break

    env.close()
    print("\nDone!")
