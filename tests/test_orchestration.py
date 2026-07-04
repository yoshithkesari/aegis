"""
LangGraph orchestration: the graph drives the controller's stages (autopilot
off) and branches at the real decision points, reaching the same outcomes as
the controller's self-drive.
"""

import tempfile

from aegis.orchestration import build_incident_graph, run_incident_via_graph
from aegis.system import build_system, run_stream


def _sys():
    return build_system(workdir=tempfile.mkdtemp(prefix="aegis_orch_"))


def test_graph_backend_is_real_langgraph_when_available():
    # langgraph is a declared dependency, so this should be the real graph.
    g = build_incident_graph(_sys().controller)
    assert g.backend == "langgraph"


def test_graph_drives_medium_incident_to_promotion():
    s = _sys()
    status = run_incident_via_graph(
        s.controller, {"drift_type": "covariate", "severity": "medium", "summary": "x"}
    )
    assert status["current_state"] == "healthy"
    inc = s.controller.incident_history[-1]
    assert inc.validation_result["passed"] is True
    assert len(s.incident_store.audit_trail(inc.incident_id)) >= 7
    assert s.controller.autopilot is True  # restored after the run


def test_graph_branches_high_severity_to_escalation():
    s = _sys()
    status = run_incident_via_graph(
        s.controller, {"drift_type": "schema", "severity": "high", "summary": "x"}
    )
    assert status["current_state"] == "escalated"  # branched away from deploy


def test_monitor_can_drive_incidents_through_the_graph():
    s = build_system(workdir=tempfile.mkdtemp(prefix="aegis_orch_"), use_graph=True)
    assert s.monitor.use_graph is True
    summary = run_stream(s, drifted=True)
    assert summary["incidents_opened"] == 1
    assert summary["final_state"] == "healthy"  # graph-driven loop closed
