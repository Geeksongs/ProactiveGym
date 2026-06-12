"""
ProactiveEnv - Gymnasium environment for training proactive agents.

This environment trains agents to:
1. Observe user activities and environment events
2. Decide when to proactively offer assistance
3. Learn from accept/reject feedback

Action Space:
    - "[predict] <task_description>": Proactively suggest a task
    - "[silent]": Stay silent, don't intervene

Observation Space:
    - events: Recent environment events
    - scenario: Current scenario description
    - goal: User's goal (what they're trying to accomplish)
    - step_count: Number of interventions so far

Reward:
    - predict + accept: +1.0 (helpful intervention)
    - predict + reject: -0.5 (false alarm, annoying)
    - silent + no_need: +0.2 (correctly stayed quiet)
    - silent + needed: -0.3 (missed opportunity)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, Any, Tuple, List, Optional

from ..config import ProactiveGymConfig, get_default_config
from ..user_profile import UserProfileGenerator
from .simulated_user import SimulatedUser
from .env_simulator import EnvironmentSimulator


class ProactiveEnv(gym.Env):
    """Gymnasium environment for training proactive agents."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Optional[ProactiveGymConfig] = None):
        """
        Initialize the ProactiveEnv.

        Args:
            config: ProactiveGymConfig with all settings
        """
        super().__init__()

        self.config = config or get_default_config()
        self.config.validate()

        # Initialize components
        self.env_simulator = EnvironmentSimulator(self.config.get_model_config())
        self.user_profile_generator = UserProfileGenerator(use_llm=True)
        self.simulated_user: Optional[SimulatedUser] = None
        self.current_user_profile = None

        # State
        self.step_count = 0
        self.episode_complete = False
        self.total_reward = 0.0
        self.action_history: List[str] = []
        self.judgment_history: List[Dict] = []

        # Spaces
        self.action_space = spaces.Text(max_length=1000)
        self.observation_space = spaces.Dict({
            "events": spaces.Text(max_length=5000),
            "scenario": spaces.Text(max_length=2000),
            "goal": spaces.Text(max_length=500),
            "feedback": spaces.Text(max_length=1000),
            "step_count": spaces.Discrete(1000),
        })

        if self.config.seed is not None:
            self.seed(self.config.seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed
            options: Additional options (can include 'theme' or 'scenario')

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset state
        self.step_count = 0
        self.episode_complete = False
        self.total_reward = 0.0
        self.action_history = []
        self.judgment_history = []

        # Get scenario
        options = options or {}
        theme = options.get("theme", random.choice(self.config.themes))

        # Generate user profile first
        self.current_user_profile = self.user_profile_generator.generate()

        if "scenario" in options:
            scenario = self.env_simulator.load_scenario(options["scenario"])
        else:
            scenario = self.env_simulator.generate_scenario(theme, self.current_user_profile)

        # Initialize simulated user with generated profile
        self.simulated_user = SimulatedUser(
            goal=scenario.user_goal,
            user_profile={
                "age": self.current_user_profile.age,
                "gender": self.current_user_profile.gender,
                "country": self.current_user_profile.country,
                "occupation": self.current_user_profile.occupation,
                "personality": self.current_user_profile.personality,
                "education_level": self.current_user_profile.education_level,
                "background": self.current_user_profile.background_description,
            },
            model_config=self.config.get_model_config(),
            reward_model_config=self.config.get_reward_model_config()
        )

        # Generate initial user activity and events
        initial_activity = self.simulated_user.generate_activity([])
        initial_events = self.env_simulator.generate_events(
            initial_activity.activity,
            max_events=self.config.max_events_per_step
        )

        # Build observation
        observation = self._build_observation(
            feedback="Episode started. Observe the user's activities and decide when to help."
        )

        info = {
            "scenario_title": scenario.title,
            "theme": scenario.theme,
            "user_goal": scenario.user_goal,
            "user_profile": {
                "age": self.current_user_profile.age,
                "gender": self.current_user_profile.gender,
                "country": self.current_user_profile.country,
                "occupation": self.current_user_profile.occupation,
                "personality": self.current_user_profile.personality,
                "education_level": self.current_user_profile.education_level,
                "background": self.current_user_profile.background_description,
            },
            "initial_events": initial_events,
        }

        if self.config.verbose:
            print(f"[ProactiveGym] New episode: {scenario.title}")
            print(f"[ProactiveGym] User: {self.current_user_profile.age}y {self.current_user_profile.gender} {self.current_user_profile.occupation} from {self.current_user_profile.country}")
            print(f"[ProactiveGym] User goal: {scenario.user_goal}")

        return observation, info

    def step(
        self,
        action_input: str
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action_input: Agent's action
                - "[predict] <task>": Suggest a task to the user
                - "[silent]": Stay silent

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.episode_complete:
            raise ValueError("Episode complete. Call reset() to start a new episode.")

        self.step_count += 1
        action_str = str(action_input).strip()
        self.action_history.append(action_str)

        # Process action
        feedback, reward, info = self._process_action(action_str)

        # Apply step penalty
        reward -= self.config.step_penalty * self.step_count

        # Normalize if configured
        if self.config.normalize_rewards:
            reward = max(-1.0, min(1.0, reward))

        self.total_reward += reward

        # Generate next user activity and events (for next observation)
        if not self.episode_complete:
            next_activity = self.simulated_user.generate_activity(
                self.env_simulator.get_recent_events()
            )
            self.env_simulator.generate_events(
                next_activity.activity,
                max_events=self.config.max_events_per_step
            )

            if self.simulated_user.is_finished:
                self.episode_complete = True

        # Check termination
        terminated = self.episode_complete
        truncated = self.step_count >= self.config.max_steps

        if terminated or truncated:
            self.episode_complete = True

        # Build observation
        observation = self._build_observation(feedback=feedback)

        info.update({
            "action": action_str,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "action_history": self.action_history.copy(),
        })

        if self.config.verbose:
            print(f"[ProactiveGym] Step {self.step_count}: {action_str[:50]}...")
            print(f"[ProactiveGym] Reward: {reward:.3f}, Total: {self.total_reward:.3f}")

        return observation, reward, terminated, truncated, info

    async def step_async(
        self,
        action_input: str
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Async version of step."""
        if self.episode_complete:
            raise ValueError("Episode complete. Call reset() to start a new episode.")

        self.step_count += 1
        action_str = str(action_input).strip()
        self.action_history.append(action_str)

        # Process action
        feedback, reward, info = await self._process_action_async(action_str)

        # Apply step penalty
        reward -= self.config.step_penalty * self.step_count

        if self.config.normalize_rewards:
            reward = max(-1.0, min(1.0, reward))

        self.total_reward += reward

        # Generate next activity and events
        if not self.episode_complete:
            next_activity = await self.simulated_user.generate_activity_async(
                self.env_simulator.get_recent_events()
            )
            await self.env_simulator.generate_events_async(
                next_activity.activity,
                max_events=self.config.max_events_per_step
            )

            if self.simulated_user.is_finished:
                self.episode_complete = True

        terminated = self.episode_complete
        truncated = self.step_count >= self.config.max_steps

        if terminated or truncated:
            self.episode_complete = True

        observation = self._build_observation(feedback=feedback)

        info.update({
            "action": action_str,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
        })

        return observation, reward, terminated, truncated, info

    def _process_action(self, action_str: str) -> Tuple[str, float, Dict]:
        """Process the agent's action and compute reward."""
        recent_events = self.env_simulator.get_recent_events()

        if action_str.startswith("[predict]"):
            # Agent is making a prediction
            predicted_task = action_str[9:].strip()

            # Get judgment from simulated user (reward model)
            judgment = self.simulated_user.judge_prediction(
                predicted_task=predicted_task,
                recent_events=recent_events
            )

            self.judgment_history.append({
                "action": "predict",
                "task": predicted_task,
                "accepted": judgment.accepted,
                "thought": judgment.thought
            })

            if judgment.accepted:
                reward = self.config.accept_reward
                feedback = f"User accepted your suggestion: '{predicted_task}'"
            else:
                reward = self.config.reject_reward
                feedback = f"User rejected your suggestion. Reason: {judgment.reason}"

            info = {
                "action_type": "predict",
                "predicted_task": predicted_task,
                "accepted": judgment.accepted,
                "judgment_thought": judgment.thought,
            }

        elif action_str.startswith("[silent]") or action_str == "[silent]":
            # Agent chose to stay silent
            needed_help, what_help = self.simulated_user.check_if_needed_help(recent_events)

            self.judgment_history.append({
                "action": "silent",
                "user_needed_help": needed_help,
                "what_help": what_help
            })

            if needed_help:
                reward = self.config.missed_opportunity_reward
                feedback = f"You stayed silent, but the user could have used help with: {what_help}"
            else:
                reward = self.config.correct_silence_reward
                feedback = "You correctly stayed silent. The user didn't need interruption."

            info = {
                "action_type": "silent",
                "user_needed_help": needed_help,
                "potential_help": what_help,
            }

        elif action_str.startswith("[finish]"):
            # Agent wants to end episode
            self.episode_complete = True
            reward = 0.0
            feedback = "Episode ended by agent."
            info = {"action_type": "finish"}

        else:
            # Invalid action format
            reward = -0.1
            feedback = "Invalid action format. Use '[predict] <task>' or '[silent]'."
            info = {"action_type": "invalid"}

        return feedback, reward, info

    async def _process_action_async(self, action_str: str) -> Tuple[str, float, Dict]:
        """Async version of _process_action."""
        recent_events = self.env_simulator.get_recent_events()

        if action_str.startswith("[predict]"):
            predicted_task = action_str[9:].strip()

            judgment = await self.simulated_user.judge_prediction_async(
                predicted_task=predicted_task,
                recent_events=recent_events
            )

            self.judgment_history.append({
                "action": "predict",
                "task": predicted_task,
                "accepted": judgment.accepted,
            })

            if judgment.accepted:
                reward = self.config.accept_reward
                feedback = f"User accepted your suggestion: '{predicted_task}'"
            else:
                reward = self.config.reject_reward
                feedback = f"User rejected your suggestion. Reason: {judgment.reason}"

            info = {
                "action_type": "predict",
                "predicted_task": predicted_task,
                "accepted": judgment.accepted,
            }

        elif action_str.startswith("[silent]") or action_str == "[silent]":
            needed_help, what_help = await self.simulated_user.check_if_needed_help_async(
                recent_events
            )

            self.judgment_history.append({
                "action": "silent",
                "user_needed_help": needed_help,
            })

            if needed_help:
                reward = self.config.missed_opportunity_reward
                feedback = f"You stayed silent, but user needed help with: {what_help}"
            else:
                reward = self.config.correct_silence_reward
                feedback = "Correctly stayed silent."

            info = {
                "action_type": "silent",
                "user_needed_help": needed_help,
            }

        elif action_str.startswith("[finish]"):
            self.episode_complete = True
            reward = 0.0
            feedback = "Episode ended."
            info = {"action_type": "finish"}

        else:
            reward = -0.1
            feedback = "Invalid action format."
            info = {"action_type": "invalid"}

        return feedback, reward, info

    def _build_observation(self, feedback: str) -> Dict[str, Any]:
        """Build observation dictionary."""
        recent_events = self.env_simulator.get_recent_events()
        events_str = "\n".join([
            f"[{e['time']}] {e['event']}" for e in recent_events
        ])

        scenario = self.env_simulator.current_scenario

        return {
            "events": events_str,
            "scenario": scenario.description if scenario else "",
            "goal": scenario.user_goal if scenario else "",
            "feedback": feedback,
            "step_count": self.step_count,
        }

    def render(self, mode: str = "human"):
        """Render the current environment state."""
        if not self.env_simulator.current_scenario:
            print("No active scenario. Call reset() first.")
            return

        scenario = self.env_simulator.current_scenario
        recent_events = self.env_simulator.get_recent_events(5)

        print("\n" + "=" * 60)
        print(f"PROACTIVE GYM - Step {self.step_count}/{self.config.max_steps}")
        print("=" * 60)
        print(f"Scenario: {scenario.title}")
        print(f"User Goal: {scenario.user_goal}")
        print(f"Total Reward: {self.total_reward:.3f}")
        print()
        print("Recent Events:")
        for e in recent_events:
            print(f"  [{e['time']}] {e['event']}")
        print()
        if self.action_history:
            print(f"Last Action: {self.action_history[-1][:80]}...")
        print("=" * 60)

    def close(self):
        """Clean up resources."""
        pass

    def seed(self, seed: int):
        """Set random seed."""
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def get_metrics(self) -> Dict[str, Any]:
        """Get episode metrics for analysis."""
        predictions = [j for j in self.judgment_history if j["action"] == "predict"]
        silences = [j for j in self.judgment_history if j["action"] == "silent"]

        accepted = sum(1 for p in predictions if p.get("accepted", False))
        rejected = len(predictions) - accepted

        correct_silence = sum(1 for s in silences if not s.get("user_needed_help", True))
        missed_opportunity = len(silences) - correct_silence

        return {
            "total_steps": self.step_count,
            "total_reward": self.total_reward,
            "predictions": len(predictions),
            "accepted": accepted,
            "rejected": rejected,
            "silences": len(silences),
            "correct_silence": correct_silence,
            "missed_opportunity": missed_opportunity,
            "precision": accepted / len(predictions) if predictions else 0.0,
            "false_alarm_rate": rejected / len(predictions) if predictions else 0.0,
        }
