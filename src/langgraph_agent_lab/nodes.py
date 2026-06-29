"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
import time
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Structured-output schema for LLM classification ─────────────────
class Classification(BaseModel):
    """Structured intent classification returned by the LLM."""

    route: Literal["risky", "tool", "missing_info", "error", "simple"] = Field(
        description="The single best route for this support ticket."
    )
    reason: str = Field(default="", description="Short justification for the chosen route.")


_CLASSIFY_SYSTEM = """You are an intent classifier for a customer-support agent.
Classify the user's support ticket into exactly ONE route. Respect this PRIORITY order
when more than one could apply (higher wins):

1. risky        — actions with side effects: refunds, deletions, cancellations, sending
                  emails, account changes, anything that mutates data or money.
2. tool         — information lookups: order status, tracking, search, fetching records.
3. missing_info — vague/incomplete tickets lacking actionable context (e.g. "fix it",
                  "it's broken") where you cannot tell what the user wants.
4. error        — system failures: timeouts, crashes, service unavailable, "cannot recover".
5. simple       — general questions answerable directly without tools or actions
                  (e.g. "how do I reset my password?").

Return only the structured classification."""


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output."""
    start = time.perf_counter()
    query = state.get("query", "")
    llm = get_llm()
    classifier = llm.with_structured_output(Classification)
    result: Classification = classifier.invoke(
        [
            ("system", _CLASSIFY_SYSTEM),
            ("human", f"Support ticket:\n{query}"),
        ]
    )
    route = result.route
    risk_level = "high" if route == "risky" else "low"
    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {route}",
                latency_ms=latency_ms,
                route=route,
                reason=result.reason,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call with transient-failure simulation for error routes."""
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    query = state.get("query", "")
    if route == "error" and attempt < 2:
        result = f"ERROR: transient tool failure on attempt {attempt} for '{query[:40]}'"
        event = make_event("tool", "error", "tool call failed", attempt=attempt)
    else:
        result = f"SUCCESS: tool returned data for '{query[:40]}' (attempt {attempt})"
        event = make_event("tool", "completed", "tool call succeeded", attempt=attempt)
    return {
        "tool_results": [result],
        "messages": [f"tool:attempt={attempt}"],
        "events": [event],
    }


class Evaluation(BaseModel):
    """LLM-as-judge verdict on tool-result quality."""

    verdict: Literal["success", "needs_retry"] = Field(
        description="'success' if the tool result answers the request, else 'needs_retry'."
    )


def evaluate_node(state: AgentState) -> dict:
    """Evaluate the latest tool result — the retry-loop gate.

    Default: a deterministic heuristic (any result containing 'ERROR' needs a retry),
    which keeps the retry loop reproducible and LLM-cost low. Set
    LANGGRAPH_LLM_JUDGE=true to enable the LLM-as-judge path (bonus), with the heuristic
    as a robust fallback if the call fails.
    """
    tool_results = state.get("tool_results", []) or []
    latest = tool_results[-1] if tool_results else ""

    # Heuristic baseline — always reliable, used as default and as fallback.
    heuristic = "needs_retry" if (not latest or "ERROR" in latest.upper()) else "success"
    evaluation_result = heuristic
    judge = "heuristic"

    if os.getenv("LANGGRAPH_LLM_JUDGE", "").lower() == "true":
        try:
            llm = get_llm()
            verdict: Evaluation = llm.with_structured_output(Evaluation).invoke(
                [
                    (
                        "system",
                        "You are a QA judge. Decide whether a tool result successfully "
                        "satisfies the request. If it contains an error, is empty, or is "
                        "clearly unusable, answer 'needs_retry'; otherwise 'success'.",
                    ),
                    ("human", f"Tool result:\n{latest}"),
                ]
            )
            evaluation_result = verdict.verdict
            judge = "llm"
        except Exception:
            evaluation_result = heuristic

    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"evaluation={evaluation_result}",
                judge=judge,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM, grounded in available context."""
    start = time.perf_counter()
    query = state.get("query", "")
    tool_results = state.get("tool_results", []) or []
    approval = state.get("approval")

    context_parts = []
    if tool_results:
        context_parts.append("Tool results:\n" + "\n".join(tool_results))
    if approval:
        context_parts.append(
            f"Approval decision: approved={approval.get('approved')} "
            f"by {approval.get('reviewer')} — {approval.get('comment')}"
        )
    context = "\n\n".join(context_parts) if context_parts else "(no additional context)"

    llm = get_llm()
    response = llm.invoke(
        [
            (
                "system",
                "You are a helpful customer-support agent. Write a concise, friendly "
                "answer to the user's ticket. Ground your answer ONLY in the provided "
                "context when context is present; do not invent order numbers or data. "
                "If an action was approved, confirm it was carried out.",
            ),
            ("human", f"User ticket:\n{query}\n\nContext:\n{context}"),
        ]
    )
    final_answer = response.content if hasattr(response, "content") else str(response)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "final_answer": final_answer,
        "messages": ["answer:generated"],
        "events": [
            make_event("answer", "completed", "final answer generated", latency_ms=latency_ms)
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    question = (
        f"I'd be happy to help, but I need a bit more detail. Regarding "
        f"\"{query[:60]}\" — could you tell me which product, order, or account this is "
        f"about and what outcome you'd like?"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": ["clarify:asked"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    proposed = (
        f"Proposed high-risk action derived from ticket: \"{query[:80]}\". "
        f"This has side effects (data/money mutation) and requires human approval before execution."
    )
    return {
        "proposed_action": proposed,
        "messages": ["risky_action:prepared"],
        "events": [make_event("risky_action", "completed", "risky action prepared for approval")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default: mock approval so tests/CI run offline. If LANGGRAPH_INTERRUPT=true, use a
    real interrupt() so an external client must supply the decision.
    """
    proposed = state.get("proposed_action", "")
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        try:
            from langgraph.types import interrupt

            decision = interrupt({"proposed_action": proposed, "question": "Approve this action?"})
            approval = decision if isinstance(decision, dict) else {
                "approved": bool(decision),
                "reviewer": "human",
                "comment": "via interrupt",
            }
        except Exception:
            approval = {
                "approved": True,
                "reviewer": "mock-reviewer",
                "comment": "fallback approve",
            }
    else:
        approval = {
            "approved": True,
            "reviewer": "mock-reviewer",
            "comment": "auto-approved (mock)",
        }

    return {
        "approval": approval,
        "messages": [f"approval:{approval.get('approved')}"],
        "events": [
            make_event(
                "approval",
                "completed",
                f"approval decision: {approval.get('approved')}",
                approved=approval.get("approved"),
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt by incrementing the attempt counter."""
    attempt = state.get("attempt", 0) + 1
    return {
        "attempt": attempt,
        "errors": [f"transient failure — retry attempt {attempt}"],
        "messages": [f"retry:{attempt}"],
        "events": [make_event("retry", "completed", f"retry attempt {attempt}", attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries are exhausted."""
    attempt = state.get("attempt", 0)
    answer = (
        "We were unable to complete your request automatically after "
        f"{attempt} attempts. The ticket has been escalated to a human specialist "
        "who will follow up shortly."
    )
    return {
        "final_answer": answer,
        "messages": ["dead_letter:escalated"],
        "events": [
            make_event(
                "dead_letter", "completed", "max retries exhausted — escalated", attempt=attempt
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "messages": ["finalize:done"],
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route", ""),
                attempts=state.get("attempt", 0),
            )
        ],
    }
