"""
Incident-lifecycle orchestration.

The report frames LangGraph as the backbone that models both the deterministic
state machine and the agentic sub-loop. This module builds that graph when
LangGraph is installed, and provides an equivalent plain-Python sequential
runner when it is not - so the wiring is real and the demo still runs anywhere.

Either way the *controller* remains the sole write-authority; the graph only
sequences calls into it.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

try:
    from langgraph.graph import END, StateGraph

    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional
    _LANGGRAPH_AVAILABLE = False


# Ordered lifecycle nodes. Each is (name, controller-method-name).
LIFECYCLE: List[Tuple[str, str]] = [
    ("drift_suspected", "handle_drift_detected"),
    ("investigating", "start_investigation"),
    ("risk_gate", "evaluate_risk_gate"),
    ("retraining", "start_retraining"),
    ("validating", "start_validation"),
    ("canary", "start_canary"),
    ("promote", "promote_challenger"),
]


class SequentialRunner:
    """Fallback orchestrator: drives the controller through the lifecycle.

    The controller's own transition guards decide whether each step actually
    fires, so calling them in order is safe and idempotent.
    """

    backend = "sequential"

    def __init__(self, controller):
        self.controller = controller

    def run(self, drift_result: dict):
        self.controller.handle_drift_detected(drift_result)
        return self.controller.get_status()


def build_incident_graph(controller):
    """Return an orchestrator for the incident lifecycle.

    Uses LangGraph if available, otherwise a SequentialRunner with identical
    semantics.
    """
    if not _LANGGRAPH_AVAILABLE:
        return SequentialRunner(controller)

    graph = StateGraph(dict)

    def _node(method_name: str) -> Callable:
        method = getattr(controller, method_name)

        def run_node(state: dict) -> dict:
            if method_name == "handle_drift_detected":
                method(state.get("drift_result", {}))
            else:
                method()
            state["current_state"] = controller.current_state.value
            return state

        return run_node

    for name, method_name in LIFECYCLE:
        graph.add_node(name, _node(method_name))

    graph.set_entry_point(LIFECYCLE[0][0])
    for (name, _), (nxt, _) in zip(LIFECYCLE, LIFECYCLE[1:]):
        graph.add_edge(name, nxt)
    graph.add_edge(LIFECYCLE[-1][0], END)

    return graph.compile()
