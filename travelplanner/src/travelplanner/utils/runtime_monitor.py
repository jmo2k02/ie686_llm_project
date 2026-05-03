"""Lightweight runtime hooks for CLI execution monitoring."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Protocol


class RunMonitor(Protocol):
    def record_llm_call(
        self, *, model_name: str, tokens_in: int, tokens_out: int
    ) -> None: ...

    def record_tool_call(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None = None,
    ) -> None: ...


_RUN_MONITOR: ContextVar[RunMonitor | None] = ContextVar(
    "travelplanner_run_monitor",
    default=None,
)


def set_run_monitor(monitor: RunMonitor | None) -> Token[RunMonitor | None]:
    return _RUN_MONITOR.set(monitor)


def reset_run_monitor(token: Token[RunMonitor | None]) -> None:
    _RUN_MONITOR.reset(token)


def get_run_monitor() -> RunMonitor | None:
    return _RUN_MONITOR.get()


def record_llm_call(*, model_name: str, tokens_in: int, tokens_out: int) -> None:
    monitor = get_run_monitor()
    if monitor is None:
        return
    monitor.record_llm_call(
        model_name=model_name,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


def record_tool_call(
    *,
    tool_name: str,
    args: dict[str, Any] | None = None,
) -> None:
    monitor = get_run_monitor()
    if monitor is None:
        return
    monitor.record_tool_call(tool_name=tool_name, args=args)
