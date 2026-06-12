"""
Environment Simulator for ProactiveGym.

This module generates environment events based on user activities.
Adapted from ProactiveAgent's EnvironmentStateManager.
"""

import json
import openai
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .prompts import build_env_events_prompt, build_scenario_generation_prompt


@dataclass
class Scenario:
    """A scenario for the proactive gym."""
    title: str
    description: str
    theme: str
    user_goal: str
    user_profile: Dict[str, Any]
    initial_state: Dict[str, Any]
    example_events: List[str]
    potential_interventions: List[str] = field(default_factory=list)


@dataclass
class EnvironmentState:
    """Current state of the environment."""
    time: str
    open_apps: List[str]
    active_window: str
    context: Dict[str, Any] = field(default_factory=dict)


class EnvironmentSimulator:
    """Simulates the environment for ProactiveGym."""

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the environment simulator.

        Args:
            model_config: LLM configuration for event generation
        """
        self.model_config = model_config
        self.current_scenario: Optional[Scenario] = None
        self.state: Optional[EnvironmentState] = None
        self.events_history: List[Dict[str, Any]] = []
        self.current_time: datetime = datetime.now()

    def load_scenario(self, scenario_data: Dict[str, Any]) -> Scenario:
        """Load a scenario from data dictionary."""
        self.current_scenario = Scenario(
            title=scenario_data.get("title", "Untitled"),
            description=scenario_data.get("description", ""),
            theme=scenario_data.get("theme", "general"),
            user_goal=scenario_data.get("user_goal", ""),
            user_profile=scenario_data.get("user_profile", {}),
            initial_state=scenario_data.get("initial_state", {}),
            example_events=scenario_data.get("example_events", []),
            potential_interventions=scenario_data.get("potential_interventions", [])
        )

        # Initialize state
        initial = self.current_scenario.initial_state
        self.state = EnvironmentState(
            time=initial.get("time", "09:00"),
            open_apps=initial.get("open_apps", []),
            active_window=initial.get("active_window", "Desktop"),
            context=initial.get("context", {})
        )

        # Parse time
        try:
            time_parts = self.state.time.split(":")
            self.current_time = datetime.now().replace(
                hour=int(time_parts[0]),
                minute=int(time_parts[1]) if len(time_parts) > 1 else 0,
                second=0
            )
        except:
            self.current_time = datetime.now()

        self.events_history = []
        return self.current_scenario

    def generate_scenario(self, theme: str) -> Scenario:
        """Generate a new scenario using LLM."""
        try:
            client = openai.OpenAI(
                api_key=self.model_config["api_key"],
                base_url=self.model_config["base_url"]
            )

            messages = build_scenario_generation_prompt(theme)

            response = client.chat.completions.create(
                model=self.model_config["model_name"],
                messages=messages,
                temperature=0.8,
                max_completion_tokens=16384,
                timeout=self.model_config.get("timeout", 60)
            )

            data = self._parse_json_response(response.choices[0].message.content)
            data["theme"] = theme
            return self.load_scenario(data)

        except Exception as e:
            print(f"[EnvironmentSimulator] Error generating scenario: {e}")
            return self._get_fallback_scenario(theme)

    async def generate_scenario_async(self, theme: str) -> Scenario:
        """Async version of generate_scenario."""
        try:
            client = openai.AsyncOpenAI(
                api_key=self.model_config["api_key"],
                base_url=self.model_config["base_url"]
            )

            messages = build_scenario_generation_prompt(theme)

            response = await client.chat.completions.create(
                model=self.model_config["model_name"],
                messages=messages,
                temperature=0.8,
                max_completion_tokens=16384,
                timeout=self.model_config.get("timeout", 60)
            )

            data = self._parse_json_response(response.choices[0].message.content)
            data["theme"] = theme
            return self.load_scenario(data)

        except Exception as e:
            print(f"[EnvironmentSimulator] Error generating scenario: {e}")
            return self._get_fallback_scenario(theme)

    def generate_events(self, activity: str, max_events: int = 5) -> List[Dict[str, Any]]:
        """
        Generate environment events based on user activity.

        Args:
            activity: Description of user's activity
            max_events: Maximum number of events to generate

        Returns:
            List of event dictionaries with 'time' and 'event' keys
        """
        try:
            client = openai.OpenAI(
                api_key=self.model_config["api_key"],
                base_url=self.model_config["base_url"]
            )

            messages = build_env_events_prompt(
                theme=self.current_scenario.theme if self.current_scenario else "general",
                activity=activity,
                recent_events=self.events_history[-5:],
                example_events=self.current_scenario.example_events if self.current_scenario else [],
                max_events=max_events
            )

            response = client.chat.completions.create(
                model=self.model_config["model_name"],
                messages=messages,
                temperature=0.7,
                max_completion_tokens=16384,
                timeout=self.model_config.get("timeout", 30)
            )

            events = self._parse_events_response(response.choices[0].message.content)

            # Add events to history
            for event in events:
                self.events_history.append(event)

            return events

        except Exception as e:
            print(f"[EnvironmentSimulator] Error generating events: {e}")
            return self._generate_fallback_events(activity)

    async def generate_events_async(self, activity: str, max_events: int = 5) -> List[Dict[str, Any]]:
        """Async version of generate_events."""
        try:
            client = openai.AsyncOpenAI(
                api_key=self.model_config["api_key"],
                base_url=self.model_config["base_url"]
            )

            messages = build_env_events_prompt(
                theme=self.current_scenario.theme if self.current_scenario else "general",
                activity=activity,
                recent_events=self.events_history[-5:],
                example_events=self.current_scenario.example_events if self.current_scenario else [],
                max_events=max_events
            )

            response = await client.chat.completions.create(
                model=self.model_config["model_name"],
                messages=messages,
                temperature=0.7,
                max_completion_tokens=16384,
                timeout=self.model_config.get("timeout", 30)
            )

            events = self._parse_events_response(response.choices[0].message.content)

            for event in events:
                self.events_history.append(event)

            return events

        except Exception as e:
            print(f"[EnvironmentSimulator] Error generating events: {e}")
            return self._generate_fallback_events(activity)

    def _parse_events_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse events from LLM response."""
        data = self._parse_json_response(response_text)
        events = []

        for event_data in data.get("events", []):
            # Update time
            time_delta = event_data.get("time_delta", random.randint(1, 30))
            self.current_time += timedelta(seconds=time_delta)

            events.append({
                "time": self.current_time.strftime("%H:%M:%S"),
                "event": event_data.get("event", "Unknown event"),
                "source": event_data.get("source", "user")
            })

        return events

    def _generate_fallback_events(self, activity: str) -> List[Dict[str, Any]]:
        """Generate fallback events when LLM fails."""
        self.current_time += timedelta(seconds=random.randint(5, 30))

        event = {
            "time": self.current_time.strftime("%H:%M:%S"),
            "event": f"The user {activity.lower()}",
            "source": "user"
        }
        self.events_history.append(event)
        return [event]

    def _get_fallback_scenario(self, theme: str) -> Scenario:
        """Get a fallback scenario when LLM fails."""
        fallback_scenarios = {
            "coding": {
                "title": "Software Development Task",
                "description": "User is working on a software development project.",
                "theme": "coding",
                "user_goal": "Implement a new feature in the codebase",
                "user_profile": {
                    "occupation": "Software Developer",
                    "expertise_level": "intermediate",
                    "working_style": "focused"
                },
                "initial_state": {
                    "time": "09:30",
                    "open_apps": ["VSCode", "Chrome", "Terminal"],
                    "active_window": "VSCode",
                    "context": {"project": "web-app"}
                },
                "example_events": [
                    "The user opens a new file in VSCode.",
                    "The user searches for documentation in Chrome.",
                    "The user runs a command in Terminal.",
                    "The user types code in the editor.",
                    "The user saves the current file."
                ]
            },
            "writing": {
                "title": "Document Writing Task",
                "description": "User is writing a document or report.",
                "theme": "writing",
                "user_goal": "Complete a written document",
                "user_profile": {
                    "occupation": "Content Writer",
                    "expertise_level": "expert",
                    "working_style": "exploratory"
                },
                "initial_state": {
                    "time": "10:00",
                    "open_apps": ["Word", "Chrome", "Notes"],
                    "active_window": "Word",
                    "context": {"document": "report.docx"}
                },
                "example_events": [
                    "The user types a paragraph in Word.",
                    "The user searches for reference material.",
                    "The user copies text from a website.",
                    "The user formats a heading.",
                    "The user saves the document."
                ]
            },
            "daily_life": {
                "title": "Daily Task Management",
                "description": "User is managing daily tasks and communications.",
                "theme": "daily_life",
                "user_goal": "Organize tasks and respond to communications",
                "user_profile": {
                    "occupation": "Office Worker",
                    "expertise_level": "intermediate",
                    "working_style": "multitasking"
                },
                "initial_state": {
                    "time": "09:00",
                    "open_apps": ["Email", "Calendar", "Browser"],
                    "active_window": "Email",
                    "context": {"unread_emails": 5}
                },
                "example_events": [
                    "The user reads an email.",
                    "The user opens the calendar app.",
                    "The user creates a new event.",
                    "The user replies to a message.",
                    "The user checks notifications."
                ]
            }
        }


        scenario_data = fallback_scenarios.get(theme, fallback_scenarios["daily_life"])
        return self.load_scenario(scenario_data)

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        text = response_text.strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {}

    def get_recent_events(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most recent events."""
        return self.events_history[-n:]

    def reset(self):
        """Reset the environment simulator."""
        self.current_scenario = None
        self.state = None
        self.events_history = []
        self.current_time = datetime.now()
