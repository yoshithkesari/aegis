"""
Incident-lifecycle orchestration.

The report frames LangGraph as the backbone that sequences the deterministic
state machine and the agentic sub-loop. This module makes that real: it drives
the controller's stages one at a time (with `autopilot` off) and branches at the
genuine decision points - risk gate -> retrain / escalate / log; validation ->
canary / rollback; canary -> promote / rollback.

When LangGraph is installed the flow is a real compiled StateGraph; otherwise an
equivalent branching runner executes the same node/router logic. Either way the
controller remains the sole write-authority - the graph only sequences calls.
"""

from __future__ import annotations

from typing import Callable, Dict

from ..control_plane.controller import IncidentState

try:
    from langgraph.graph import END, StateGraph

    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional
    _LANGGRAPH_AVAILABLE = False
    END = "__end__"

# Ordered stage nodes and the controller method each one runs.
NODES = ["investigate", "risk_gate", "retrain", "validate", "canary", "promote"]


def _run_node(controller, name: str) -> None:
    """Execute exactly one controller stage (no auto-chaining)."""
    if name == "investigate":
        controller.start_investigation()
    elif name == "risk_gate":
        controller.evaluate_risk_gate()
    elif name == "retrain":
        controller.start_retraining(controller._pending_shadow_hold)
    elif name == "validate":
        controller.start_validation(controller._pending_shadow_hold)
    elif name == "canary":
        controller.start_canary()
    elif name == "promote":
        controller.promote_challenger()


def _route(controller, name: str):
    """Decide the next node from the controller's resulting state (or END)."""
    s = controller.current_state
    if name == "investigate":
        return "risk_gate" if s is IncidentState.DIAGNOSED else END
    if name == "risk_gate":
        return "retrain" if s is IncidentState.DIAGNOSED else END
    if name == "retrain":
        return "validate" if s is IncidentState.RETRAINING else END
    if name == "validate":
        return "canary" if s is IncidentState.VALIDATING else END
    if name == "canary":
        return "promote" if s is IncidentState.CANARY else END
    return END  # promote is terminal


# Possible successors per node, for building conditional edges.
_TARGETS: Dict[str, list] = {
    "investigate": ["risk_gate"],
    "risk_gate": ["retrain"],
    "retrain": ["validate"],
    "validate": ["canary"],
    "canary": ["promote"],
    "promote": [],
}


class IncidentGraph:
    """Runs the incident lifecycle by sequencing controller stages.

    Assumes the incident is already open (controller in DRIFT_SUSPECTED with
    autopilot off); `run()` drives it to a terminal state.
    """

    def __init__(self, controller, compiled=None):
        self.controller = controller
        self._compiled = compiled
        self.backend = "langgraph" if compiled is not None else "sequential"

    def run(self) -> Dict:
        if self._compiled is not None:
            self._compiled.invoke({})
        else:
            node = NODES[0]
            while node != END:
                _run_node(self.controller, node)
                node = _route(self.controller, node)
        return self.controller.get_status()


def build_incident_graph(controller) -> IncidentGraph:
    """Build the lifecycle orchestrator (real LangGraph when available)."""
    if not _LANGGRAPH_AVAILABLE:
        return IncidentGraph(controller)

    graph = StateGraph(dict)

    def make_node(name: str) -> Callable:
        def node(state: dict) -> dict:
            _run_node(controller, name)
            return state
        return node

    for name in NODES:
        graph.add_node(name, make_node(name))

    graph.set_entry_point(NODES[0])
    for name in NODES:
        router = (lambda n: (lambda state: _route(controller, n)))(name)
        path_map = {t: t for t in _TARGETS[name]}
        path_map[END] = END
        graph.add_conditional_edges(name, router, path_map)

    return IncidentGraph(controller, compiled=graph.compile())


def run_incident_via_graph(controller, drift_result: Dict) -> Dict:
    """Open an incident and drive it to resolution via the graph orchestrator."""
    prev = controller.autopilot
    controller.autopilot = False
    try:
        controller.handle_drift_detected(drift_result)  # opens, stops at DRIFT_SUSPECTED
        build_incident_graph(controller).run()
    finally:
        controller.autopilot = prev
    return controller.get_status()
