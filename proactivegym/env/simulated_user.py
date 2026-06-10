"""
Simulated User for ProactiveGym.

This module simulates a user who:
1. Generates activities as they work toward a goal
2. Judges whether agent's proactive suggestions should be accepted/rejected

Adapted from ProactiveAgent's UserAgent and RewardModel components.
"""

import json
import openai
import asyncio
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from .prompts import (
    build_user_activity_prompt,
    build_reward_model_prompt,
    build_user_need_help_prompt,
)


@dataclass
class UserJudgment:
    """Result of user judging an agent's prediction."""
    accepted: bool
    thought: str
    reason: str


@dataclass
class UserActivity:
    """User's next activity."""
    thought: str
    activity: str
    is_finished: bool
    needs_help: bool = False


class SimulatedUser:
    """Simulates a user for the ProactiveGym environment."""

    def __init__(
        self,
        goal: str,
        user_profile: Dict[str, Any],
        model_config: Dict[str, Any],
        reward_model_config: Dict[str, Any],
    ):
        """
        Initialize the simulated user.

        Args:
            goal: What the user is trying to accomplish
            user_profile: User characteristics (occupation, expertise, etc.)
            model_config: LLM config for activity generation
            reward_model_config: LLM config for judgment (reward model)
        """
        self.goal = goal
        self.user_profile = user_profile
        self.model_config = model_config
        self.reward_model_config = reward_model_config

        # State
        self.activity_history: List[str] = []
        self.is_finished = False

        # Build user info string
        self.user_info = self._build_user_info()

    def _build_user_info(self) -> str:
        """Build user info string from profile."""
        lines = []
        for key, value in self.user_profile.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def generate_activity(self, recent_events: List[Dict]) -> UserActivity:
        """
        Generate the user's next activity based on recent events.

        Args:
            recent_events: List of recent environment events

        Returns:
            UserActivity with the user's next action
        """
        try:
            client = openai.OpenAI(
                api_key=self.model_config["api_key"],
                base_url=self.model_config["base_url"]
            )

            messages = build_user_activity_prompt(
                goal=self.goal,
                user_info=self.user_info,
                recent_events=recent_events
            )

            response = client.chat.completions.create(
                model=self.model_config["model_name"],
                messages=messages,
                temperature=self.model_config.get("temperature", 0.7),
                max_tokens=self.model_config.get("max_tokens", 1024),
                timeout=self.model_config.get("timeout", 30)
            )

            result = self._parse_activity_response(response.choices[0].message.content)
            self.activity_history.append(result.activity)

            if result.is_finished:
                self.is_finished = True

            return result

        except Exception as e:
            print(f"[SimulatedUser] Error generating activity: {e}")
            return UserActivity(
                thought="Continuing with the task",
                activity="The user continues working on the current task.",
                is_finished=False
            )

    async def generate_activity_async(self, recent_events: List[Dict]) -> UserActivity:
        """Async version of generate_activity."""
        try:
            client = openai.AsyncOpenAI(
                api_key=self.model_config["api_key"],
                base_url=self.model_config["base_url"]
            )

            messages = build_user_activity_prompt(
                goal=self.goal,
                user_info=self.user_info,
                recent_events=recent_events
            )

            response = await client.chat.completions.create(
                model=self.model_config["model_name"],
                messages=messages,
                temperature=self.model_config.get("temperature", 0.7),
                max_tokens=self.model_config.get("max_tokens", 1024),
                timeout=self.model_config.get("timeout", 30)
            )

            result = self._parse_activity_response(response.choices[0].message.content)
            self.activity_history.append(result.activity)

            if result.is_finished:
                self.is_finished = True

            return result

        except Exception as e:
            print(f"[SimulatedUser] Error generating activity: {e}")
            return UserActivity(
                thought="Continuing with the task",
                activity="The user continues working on the current task.",
                is_finished=False
            )

    def judge_prediction(
        self,
        predicted_task: Optional[str],
        recent_events: List[Dict]
    ) -> UserJudgment:
        """
        Judge whether to accept or reject the agent's prediction.

        This uses the Reward Model approach from ProactiveAgent (F1=91.8%).

        Args:
            predicted_task: The task predicted by agent, or None for [silent]
            recent_events: Recent environment events

        Returns:
            UserJudgment with accept/reject decision
        """
        try:
            client = openai.OpenAI(
                api_key=self.reward_model_config["api_key"],
                base_url=self.reward_model_config["base_url"]
            )

            messages = build_reward_model_prompt(
                events=recent_events,
                predicted_task=predicted_task
            )

            response = client.chat.completions.create(
                model=self.reward_model_config["model_name"],
                messages=messages,
                temperature=self.reward_model_config.get("temperature", 0.0),
                max_tokens=self.reward_model_config.get("max_tokens", 1024),
                timeout=self.reward_model_config.get("timeout", 30)
            )

            return self._parse_judgment_response(response.choices[0].message.content)

        except Exception as e:
            print(f"[SimulatedUser] Error judging prediction: {e}")
            # Default to reject on error to avoid false positives
            return UserJudgment(
                accepted=False,
                thought="Error in judgment",
                reason=str(e)
            )

    async def judge_prediction_async(
        self,
        predicted_task: Optional[str],
        recent_events: List[Dict]
    ) -> UserJudgment:
        """Async version of judge_prediction."""
        try:
            client = openai.AsyncOpenAI(
                api_key=self.reward_model_config["api_key"],
                base_url=self.reward_model_config["base_url"]
            )

            messages = build_reward_model_prompt(
                events=recent_events,
                predicted_task=predicted_task
            )

            response = await client.chat.completions.create(
                model=self.reward_model_config["model_name"],
                messages=messages,
                temperature=self.reward_model_config.get("temperature", 0.0),
                max_tokens=self.reward_model_config.get("max_tokens", 1024),
                timeout=self.reward_model_config.get("timeout", 30)
            )

            return self._parse_judgment_response(response.choices[0].message.content)

        except Exception as e:
            print(f"[SimulatedUser] Error judging prediction: {e}")
            return UserJudgment(
                accepted=False,
                thought="Error in judgment",
                reason=str(e)
            )

    def check_if_needed_help(self, recent_events: List[Dict]) -> Tuple[bool, str]:
        """
        Check if the user needed help at this moment (for evaluating [silent] action).

        Args:
            recent_events: Recent environment events

        Returns:
            Tuple of (needed_help: bool, reason: str)
        """
        try:
            client = openai.OpenAI(
                api_key=self.reward_model_config["api_key"],
                base_url=self.reward_model_config["base_url"]
            )

            messages = build_user_need_help_prompt(recent_events)

            response = client.chat.completions.create(
                model=self.reward_model_config["model_name"],
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                timeout=self.reward_model_config.get("timeout", 30)
            )

            result = self._parse_json_response(response.choices[0].message.content)
            return result.get("user_needed_help", False), result.get("what_help", "")

        except Exception as e:
            print(f"[SimulatedUser] Error checking if needed help: {e}")
            return False, ""

    async def check_if_needed_help_async(self, recent_events: List[Dict]) -> Tuple[bool, str]:
        """Async version of check_if_needed_help."""
        try:
            client = openai.AsyncOpenAI(
                api_key=self.reward_model_config["api_key"],
                base_url=self.reward_model_config["base_url"]
            )

            messages = build_user_need_help_prompt(recent_events)

            response = await client.chat.completions.create(
                model=self.reward_model_config["model_name"],
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                timeout=self.reward_model_config.get("timeout", 30)
            )

            result = self._parse_json_response(response.choices[0].message.content)
            return result.get("user_needed_help", False), result.get("what_help", "")

        except Exception as e:
            print(f"[SimulatedUser] Error checking if needed help: {e}")
            return False, ""

    def _parse_activity_response(self, response_text: str) -> UserActivity:
        """Parse LLM response for activity generation."""
        data = self._parse_json_response(response_text)
        return UserActivity(
            thought=data.get("thought", ""),
            activity=data.get("activity", "The user continues working."),
            is_finished=data.get("is_finished", False),
            needs_help=data.get("needs_help", False)
        )

    def _parse_judgment_response(self, response_text: str) -> UserJudgment:
        """Parse LLM response for judgment."""
        data = self._parse_json_response(response_text)
        judgment_str = data.get("judgment", "rejected").lower()
        return UserJudgment(
            accepted=(judgment_str == "accepted"),
            thought=data.get("thought", ""),
            reason=data.get("reason", "")
        )

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {}

    def reset(self, goal: str, user_profile: Dict[str, Any]):
        """Reset the simulated user for a new episode."""
        self.goal = goal
        self.user_profile = user_profile
        self.user_info = self._build_user_info()
        self.activity_history = []
        self.is_finished = False
