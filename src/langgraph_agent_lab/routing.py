"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from .state import AgentState, Route

# Maps the LLM-classified route to the next graph node.
_CLASSIFY_TARGETS: dict[str, str] = {
    Route.SIMPLE.value: "answer",
    Route.TOOL.value: "tool",
    Route.MISSING_INFO.value: "clarify",
    Route.RISKY.value: "risky_action",
    Route.ERROR.value: "retry",
}


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node (defaults to ``answer``)."""
    return _CLASSIFY_TARGETS.get(state.get("route", ""), "answer")


def route_after_evaluate(state: AgentState) -> str:
    """Decide if the tool result is satisfactory or needs another attempt.

    This is the 'done?' check that creates the retry loop.
    """
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry the tool or give up — MUST be bounded."""
    if state.get("attempt", 0) < state.get("max_attempts", 3):
        return "tool"
    return "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Route based on the human approval decision."""
    approval = state.get("approval") or {}
    if approval.get("approved"):
        return "tool"
    return "clarify"
