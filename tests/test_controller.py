"""
Controller state-machine tests.

Uses lightweight fakes for the injected dependencies so we exercise the
deterministic transition logic (not the ML). These guard the bugs that were
just fixed: the DIAGNOSED transition and the retrain->validate hand-off.
"""

from aegis.agent.investigator import InvestigationResult
from aegis.control_plane.controller import Controller, IncidentState
from aegis.control_plane.risk_gate import RiskGate


class FakeInvestigator:
    def investigate(self, incident):
        return InvestigationResult(
            diagnosis="concept drift in merchant=X",
            root_cause="distribution shift in transaction_amount",
            recommended_action="retrain challenger",
            confidence=0.8,
            tool_calls={},
            reasoning="fake",
        )


class FakeRemediation:
    def retrain(self, incident):
        return {"success": True, "job_id": "job-1"}

    def deploy_canary(self, incident):
        return {"success": True, "metrics": {"accuracy": 0.89}}

    def promote(self, incident):
        return {"success": True}

    def rollback(self, incident):
        return {"success": True}


class FakeValidator:
    def validate(self, incident, label_free=False):
        return {"passed": True}


def _wired_controller():
    c = Controller(model_id="fraud-classifier")
    c.set_dependencies(
        detectors=None,
        risk_gate=RiskGate(),
        remediation=FakeRemediation(),
        investigator=FakeInvestigator(),
        validator=FakeValidator(),
    )
    return c


def test_medium_drift_runs_full_loop_and_resolves():
    c = _wired_controller()
    c.handle_drift_detected({"drift_type": "concept", "severity": "medium", "summary": "x"})

    # loop closed: promotion resets to HEALTHY and clears the incident
    assert c.current_state is IncidentState.HEALTHY
    assert c.current_incident is None
    assert len(c.incident_history) == 1


def test_diagnosis_is_recorded_on_incident():
    c = _wired_controller()
    # capture the incident before it is cleared on resolution
    seen = {}
    orig = c.transition_to

    def spy(state, reason=""):
        if state is IncidentState.DIAGNOSED and c.current_incident:
            seen["root_cause"] = c.current_incident.root_cause
        return orig(state, reason)

    c.transition_to = spy
    c.handle_drift_detected({"drift_type": "concept", "severity": "medium"})
    assert seen.get("root_cause") == "distribution shift in transaction_amount"


def test_high_severity_escalates_and_does_not_deploy():
    c = _wired_controller()
    c.handle_drift_detected({"drift_type": "schema", "severity": "high"})
    assert c.current_state is IncidentState.ESCALATED
    assert c.current_incident is not None  # left open for a human


def test_drift_ignored_when_not_healthy_is_idempotent():
    c = _wired_controller()
    c.current_state = IncidentState.ESCALATED
    before = len(c.incident_history)
    c.handle_drift_detected({"drift_type": "concept", "severity": "medium"})
    assert len(c.incident_history) == before  # no new incident opened
