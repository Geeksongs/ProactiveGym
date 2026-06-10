"""
Prompts for ProactiveGym components.

These prompts are adapted from ProactiveAgent (ICLR 2025) with proven effectiveness:
- Reward Model F1-Score: 91.8%
- Agent Model outperforms GPT-4o
"""

import json
from typing import Dict, Any, List


# ============================================================================
# SIMULATED USER - Activity Generation
# Adapted from: ProactiveAgent/gym/components/user.py
# ============================================================================

USER_SYSTEM_PROMPT = """<Role>
You are tasked with simulating a user within a system. The content labeled `Source: user` reflects your past actions and decisions.
</Role>

<Task>
Generate human-like activities with distinct characteristics and identities. You will receive events and observations from the environment; analyze these closely to decide your actions.
</Task>

<Rules>
- Respond like a real user; don't be overly predictable.
- Refer to # User Info to understand your identity.
- Critically evaluate the received information, as it may not always be accurate.
- Stay aware of environmental changes, which can occur at any time.
</Rules>"""


def build_user_activity_prompt(goal: str, user_info: str, recent_events: List[Dict]) -> List[Dict]:
    """Build prompt for generating user's next activity."""
    events_str = "\n".join([f"- [{e['time']}] {e['event']}" for e in recent_events[-10:]])

    messages = [
        {"role": "system", "content": USER_SYSTEM_PROMPT},
        {"role": "user", "content": f"# Goal\n{goal}\n\n# User Info\n{user_info}"},
        {"role": "user", "content": f"""# Recent Events
{events_str}

Now describe what's your next action to achieve the goal based on the environmental observation.

Respond in JSON format:
{{
    "thought": "What you're thinking",
    "activity": "Your specific action",
    "is_finished": false
}}"""}
    ]
    return messages


# ============================================================================
# REWARD MODEL - Judge Accept/Reject
# Adapted from: ProactiveAgent/eval/reward_model_template.py (F1=91.8%)
# ============================================================================

REWARD_MODEL_SYSTEM_PROMPT = """<Task>
Evaluate the task proposed by the proactive assistant as the user.
</Task>

<Rule>
0. Analyze the current observation to understand your current situation and requirements.
1. If the proposed task is 'null' (indicating no task is proposed under the current observation), follow these steps:
   - Accept the 'null' task if you believe there is no need for a task.
   - Reject the 'null' task if you believe a task is needed.
2. Minimize interruptions from the assistant by only accepting tasks that are valuable.
3. Evaluate the current observation and make a judgment on the proposed task accordingly.
</Rule>

<Format>
You should answer with the following JSON format:
{
    "thought": "Give your thoughts first, then provide the judgment of the task.",
    "judgment": "accepted or rejected"
}
</Format>"""


def build_reward_model_prompt(events: List[Dict], predicted_task: str) -> List[Dict]:
    """Build prompt for reward model to judge accept/reject."""
    obs_list = [{"time": e["time"], "event": e["event"]} for e in events[-10:]]

    user_content = {
        "Observations (Time Ascending)": obs_list,
        "Proposed Task": predicted_task,
        "Instruction": "Now give your judgment. You should complete the reasoning process in the first person."
    }

    messages = [
        {"role": "system", "content": REWARD_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_content, indent=2)}
    ]
    return messages


# ============================================================================
# CHECK IF USER NEEDED HELP (for silent action evaluation)
# ============================================================================

USER_NEED_HELP_PROMPT = """<Task>
Analyze the user's recent activities to determine if they needed assistance.
</Task>

<Observations>
{events}
</Observations>

<Rule>
Consider whether the user was:
- Struggling or repeating actions (might need help)
- Searching for information they couldn't find (might need help)
- In a productive flow (should NOT be interrupted)
- Doing routine tasks (no help needed)
</Rule>

<Format>
{{
    "thought": "Analysis of user's situation",
    "user_needed_help": true or false,
    "what_help": "Description of what help would be useful (if any)"
}}
</Format>"""


def build_user_need_help_prompt(events: List[Dict]) -> List[Dict]:
    """Build prompt to check if user needed help (for evaluating silent action)."""
    events_str = "\n".join([f"- [{e['time']}] {e['event']}" for e in events[-10:]])

    messages = [
        {"role": "user", "content": USER_NEED_HELP_PROMPT.format(events=events_str)}
    ]
    return messages


# ============================================================================
# ENVIRONMENT SIMULATOR - Event Generation
# Adapted from: ProactiveAgent/gym/components/environment.py
# ============================================================================

ENV_SYSTEM_PROMPT = """<Role>
You are tasked with simulating an environment within a system. The content labeled `Source: environment` reflects your past actions and decisions.
</Role>

<Task>
Generate and refine detailed environment settings. Based on the latest activities, create multiple events to describe changes in the environment.
</Task>

<Rules>
- Ensure the subject of the generated content aligns with the latest activities's source.
- Avoid subjective opinions or emotions; focus on objective changes.
- Ensure events are consistent with historical events labeled `[events]` and include all changes from the activities.
- Introduce occasional failures or unexpected events for realism.
- Ensure each event is logically connected to the previous one and does not include nonexistent elements.
- Pay close attention to entity operations; if an operation is not allowed or impractical in the real or simulated environment, raise an error and explain the issue.
</Rules>"""


def build_env_events_prompt(
    theme: str,
    activity: str,
    recent_events: List[Dict],
    example_events: List[str],
    max_events: int = 5
) -> List[Dict]:
    """Build prompt for generating environment events from user activity."""
    recent_str = "\n".join([f"- [{e['time']}] {e['event']}" for e in recent_events[-5:]])
    examples_str = "\n".join([f"- {e}" for e in example_events[:5]])

    prompt = f"""<Theme>
{theme}
</Theme>

<Recent Events>
{recent_str}
</Recent Events>

<Example Events (for style reference)>
{examples_str}
</Example Events>

<User Activity>
{activity}
</User Activity>

Based on the user's activity, generate 1-{max_events} events describing environmental changes.
Make sure the events' subject is the User.

Respond in JSON format:
{{
    "events": [
        {{"time_delta": <seconds>, "event": "description"}},
        ...
    ]
}}"""

    messages = [
        {"role": "system", "content": ENV_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    return messages


# ============================================================================
# SCENARIO GENERATION
# ============================================================================

SCENARIO_GENERATION_PROMPT = """Generate a realistic scenario for proactive agent training.

Theme: {theme}

Create a scenario where a user is working on a task and might benefit from proactive AI assistance at various points.

Respond in JSON format:
{{
    "title": "Brief scenario title",
    "description": "Detailed scenario description",
    "user_goal": "What the user is trying to accomplish",
    "user_profile": {{
        "occupation": "...",
        "expertise_level": "beginner/intermediate/expert",
        "working_style": "focused/multitasking/exploratory"
    }},
    "initial_state": {{
        "time": "HH:MM",
        "open_apps": ["app1", "app2"],
        "active_window": "...",
        "context": "..."
    }},
    "example_events": [
        "Example event 1",
        "Example event 2",
        "Example event 3"
    ],
    "potential_interventions": [
        "When user might need help 1",
        "When user might need help 2"
    ]
}}"""


def build_scenario_generation_prompt(theme: str) -> List[Dict]:
    """Build prompt for generating a new scenario."""
    messages = [
        {"role": "user", "content": SCENARIO_GENERATION_PROMPT.format(theme=theme)}
    ]
    return messages
