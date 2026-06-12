import os
from dataclasses import dataclass, field
from typing import Optional, Union, List


@dataclass
class ProactiveGymConfig:
    """Configuration class for ProactiveGym environment.

    This gym trains agents to proactively assist users by:
    1. Observing user activities and environment events
    2. Deciding when to intervene (timing)
    3. Predicting what help the user needs (task prediction)
    4. Learning from accept/reject feedback
    """

    # Model configuration (for simulated user / reward model)
    api_key: str = ""
    model_name: str = "gpt-5.4-nano-2026-03-17"
    base_url: str = ""
    temperature: float = 0.7
    max_completion_tokens: int = 16384
    timeout: int = 30

    # Reward Model configuration
    reward_model_name: str = "gpt-5.4-nano-2026-03-17"  # More capable model for judgment
    reward_model_temperature: float = 0.0  # Deterministic judgment

    # Environment configuration
    max_steps: int = 150  # Max interventions per episode
    max_events_per_step: int = 5  # Events generated between agent decisions
    verbose: bool = False
    seed: Optional[int] = None

    # Reward configuration
    accept_reward: float = 1.0       # Agent predicts, user accepts
    reject_reward: float = -0.1      # Agent predicts, user rejects (false alarm)
    correct_silence_reward: float = 0.001   # Agent silent, user didn't need help
    missed_opportunity_reward: float = -0.3  # Agent silent, user needed help
    step_penalty: float = 0.0
    normalize_rewards: bool = False

    # Data configuration
    data_mode: str = "random"  # "random", "single", "list"
    data_source: Optional[Union[str, List[str]]] = None
    scenarios_path: Optional[str] = None  # Path to scenarios YAML file

    # Scenario themes
    themes: List[str] = field(default_factory=lambda: ["coding", "writing", "daily_life"])

    def __post_init__(self):
        """Post-initialization setup."""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def validate(self):
        """Validate configuration parameters."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_events_per_step <= 0:
            raise ValueError("max_events_per_step must be positive")
        if self.data_mode not in ["random", "single", "list"]:
            raise ValueError("data_mode must be 'random', 'single', or 'list'")
        if not self.api_key:
            raise ValueError("api_key is required. Set OPENAI_API_KEY environment variable.")
        return True

    def get_model_config(self) -> dict:
        """Get model configuration as a dictionary."""
        return {
            "api_key": self.api_key,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "timeout": self.timeout,
        }

    def get_reward_model_config(self) -> dict:
        """Get reward model configuration as a dictionary."""
        return {
            "api_key": self.api_key,
            "model_name": self.reward_model_name,
            "base_url": self.base_url,
            "temperature": self.reward_model_temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "timeout": self.timeout,
        }


def get_default_config() -> ProactiveGymConfig:
    """Get default configuration."""
    return ProactiveGymConfig()


def get_demo_config() -> ProactiveGymConfig:
    """Get configuration optimized for demos."""
    return ProactiveGymConfig(
        verbose=True,
        max_steps=10,
        max_events_per_step=3,
        step_penalty=0.0,
    )


def get_training_config() -> ProactiveGymConfig:
    """Get configuration optimized for RL training."""
    return ProactiveGymConfig(
        verbose=False,
        max_steps=150,
        max_events_per_step=5,
        normalize_rewards=False,
        step_penalty=0.0,
    )
