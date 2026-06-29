# Day 08 Lab Report — LangGraph Support-Ticket Agent

## 1. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100% |
| Avg nodes visited | 6.43 |
| Total retries | 3 |
| Total interrupts (approvals) | 2 |
| Resume success | False |

## 2. Per-scenario results

| Scenario | Expected | Actual | Success | Retries | Interrupts | Approval req/obs |
|---|---|---|:---:|---:|---:|:---:|
| S01_simple | simple | simple | ✅ | 0 | 0 | False/False |
| S02_tool | tool | tool | ✅ | 0 | 0 | False/False |
| S03_missing | missing_info | missing_info | ✅ | 0 | 0 | False/False |
| S04_risky | risky | risky | ✅ | 0 | 1 | True/True |
| S05_error | error | error | ✅ | 2 | 0 | False/False |
| S06_delete | risky | risky | ✅ | 0 | 1 | True/True |
| S07_dead_letter | error | error | ✅ | 1 | 0 | False/False |

## 3. Architecture

The graph is a `StateGraph(AgentState)` with 11 nodes. Flow:

```
START → intake → classify → (route_after_classify)
  simple       → answer → finalize → END
  tool         → tool → evaluate → (route_after_evaluate)
                                    success     → answer → finalize → END
                                    needs_retry → retry → (route_after_retry)
                                                           tool (retry) | dead_letter → finalize → END
  missing_info → clarify → finalize → END
  risky        → risky_action → approval → (route_after_approval)
                                            approved → tool → evaluate → ...
                                            rejected → clarify → finalize → END
  error        → retry → (route_after_retry) → ...
```

`classify_node` and `answer_node` make real LLM calls; `classify_node` uses `.with_structured_output(Classification)` for a reliable enum route, and `evaluate_node` uses an LLM-as-judge with a heuristic fallback.

## 4. State schema

| Field | Reducer | Why |
|---|---|---|
| messages | append (`operator.add`) | running audit trail |
| tool_results | append | accumulate each tool attempt |
| errors | append | accumulate transient failures |
| events | append | append-only audit events for grading |
| route / risk_level | overwrite | only the current classification matters |
| attempt | overwrite | latest retry counter |
| evaluation_result | overwrite | gates the retry loop |
| pending_question | overwrite | clarification question |
| proposed_action | overwrite | risky action awaiting approval |
| approval | overwrite | latest HITL decision |

## 5. Failure analysis

1. **Transient tool failure / unbounded retry** — `tool_node` simulates errors for the `error` route. `evaluate_node` flags `needs_retry`, and `route_after_retry` is **bounded** by `attempt < max_attempts`; once exhausted it routes to `dead_letter`, which escalates instead of looping forever (see S07 with `max_attempts=1`).
2. **Risky action without approval** — `risky` queries cannot reach `tool` directly; they must pass through `risky_action → approval`, and only `route_after_approval` (approved) lets execution continue. Rejected actions divert to `clarify`.

## 6. Persistence / recovery

Each scenario runs with a unique `thread_id` (`thread-<scenario_id>`). With the `sqlite` checkpointer, state is persisted to `outputs/checkpoints.db` (WAL mode), so `graph.get_state_history(config)` can replay every step and a run can resume after a process crash.

## 7. Improvement plan

- Real HITL via `interrupt()` + a Streamlit approve/reject UI.
- Parallel tool fan-out with `Send()` for multi-lookup tickets.
- Replace the mock tool with real integrations and add per-node latency budgets.
- Add caching for classification to cut LLM cost on repeated tickets.
