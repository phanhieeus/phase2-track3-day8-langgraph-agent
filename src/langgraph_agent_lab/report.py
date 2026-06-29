"""Report generation helper."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report (markdown) from metrics data."""
    lines: list[str] = []
    lines.append("# Day 08 Lab Report — LangGraph Support-Ticket Agent\n")

    # 1. Summary ------------------------------------------------------
    lines.append("## 1. Metrics summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total scenarios | {metrics.total_scenarios} |")
    lines.append(f"| Success rate | {metrics.success_rate:.0%} |")
    lines.append(f"| Avg nodes visited | {metrics.avg_nodes_visited:.2f} |")
    lines.append(f"| Total retries | {metrics.total_retries} |")
    lines.append(f"| Total interrupts (approvals) | {metrics.total_interrupts} |")
    lines.append(f"| Resume success | {metrics.resume_success} |\n")

    # 2. Per-scenario -------------------------------------------------
    lines.append("## 2. Per-scenario results\n")
    lines.append("| Scenario | Expected | Actual | Success | Retries | Interrupts | Approval req/obs |")
    lines.append("|---|---|---|:---:|---:|---:|:---:|")
    for m in metrics.scenario_metrics:
        ok = "✅" if m.success else "❌"
        lines.append(
            f"| {m.scenario_id} | {m.expected_route} | {m.actual_route} | {ok} | "
            f"{m.retry_count} | {m.interrupt_count} | {m.approval_required}/{m.approval_observed} |"
        )
    lines.append("")

    # 3. Architecture -------------------------------------------------
    lines.append("## 3. Architecture\n")
    lines.append(
        "The graph is a `StateGraph(AgentState)` with 11 nodes. Flow:\n\n"
        "```\n"
        "START → intake → classify → (route_after_classify)\n"
        "  simple       → answer → finalize → END\n"
        "  tool         → tool → evaluate → (route_after_evaluate)\n"
        "                                    success     → answer → finalize → END\n"
        "                                    needs_retry → retry → (route_after_retry)\n"
        "                                                           tool (retry) | dead_letter → finalize → END\n"
        "  missing_info → clarify → finalize → END\n"
        "  risky        → risky_action → approval → (route_after_approval)\n"
        "                                            approved → tool → evaluate → ...\n"
        "                                            rejected → clarify → finalize → END\n"
        "  error        → retry → (route_after_retry) → ...\n"
        "```\n\n"
        "`classify_node` and `answer_node` make real LLM calls; `classify_node` uses "
        "`.with_structured_output(Classification)` for a reliable enum route, and "
        "`evaluate_node` uses an LLM-as-judge with a heuristic fallback.\n"
    )

    # 4. State schema -------------------------------------------------
    lines.append("## 4. State schema\n")
    lines.append("| Field | Reducer | Why |")
    lines.append("|---|---|---|")
    lines.append("| messages | append (`operator.add`) | running audit trail |")
    lines.append("| tool_results | append | accumulate each tool attempt |")
    lines.append("| errors | append | accumulate transient failures |")
    lines.append("| events | append | append-only audit events for grading |")
    lines.append("| route / risk_level | overwrite | only the current classification matters |")
    lines.append("| attempt | overwrite | latest retry counter |")
    lines.append("| evaluation_result | overwrite | gates the retry loop |")
    lines.append("| pending_question | overwrite | clarification question |")
    lines.append("| proposed_action | overwrite | risky action awaiting approval |")
    lines.append("| approval | overwrite | latest HITL decision |\n")

    # 5. Failure analysis --------------------------------------------
    lines.append("## 5. Failure analysis\n")
    lines.append(
        "1. **Transient tool failure / unbounded retry** — `tool_node` simulates errors for "
        "the `error` route. `evaluate_node` flags `needs_retry`, and `route_after_retry` is "
        "**bounded** by `attempt < max_attempts`; once exhausted it routes to `dead_letter`, "
        "which escalates instead of looping forever (see S07 with `max_attempts=1`).\n"
        "2. **Risky action without approval** — `risky` queries cannot reach `tool` directly; "
        "they must pass through `risky_action → approval`, and only `route_after_approval` "
        "(approved) lets execution continue. Rejected actions divert to `clarify`.\n"
    )

    # 6. Persistence --------------------------------------------------
    lines.append("## 6. Persistence / recovery\n")
    lines.append(
        "Each scenario runs with a unique `thread_id` (`thread-<scenario_id>`). With the "
        "`sqlite` checkpointer, state is persisted to `outputs/checkpoints.db` (WAL mode), so "
        "`graph.get_state_history(config)` can replay every step and a run can resume after a "
        "process crash.\n"
    )

    # 7. Improvement plan --------------------------------------------
    lines.append("## 7. Improvement plan\n")
    lines.append(
        "- Real HITL via `interrupt()` + a Streamlit approve/reject UI.\n"
        "- Parallel tool fan-out with `Send()` for multi-lookup tickets.\n"
        "- Replace the mock tool with real integrations and add per-node latency budgets.\n"
        "- Add caching for classification to cut LLM cost on repeated tickets.\n"
    )

    return "\n".join(lines)


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
