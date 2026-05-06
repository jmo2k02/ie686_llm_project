"""Message history helpers for planner graph nodes."""

from __future__ import annotations

from travelplanner.schema.system_state import MessageHistoryModel


def build_message_history(
    *,
    user_agent: str,
    agent_ref: str,
    query: str,
    user_prompt: str,
    raw_response: str,
) -> MessageHistoryModel:
    return MessageHistoryModel(
        user_agent=user_agent,
        model="llm",
        agent_ref=agent_ref,
        messages=[
            {"role": "user", "content": query},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": raw_response},
        ],
    )
