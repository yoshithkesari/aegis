"""
Stream -> detector -> auto-incident loop.

The monitor must stay silent on same-distribution data (no false interventions),
and auto-open + auto-resolve an incident when real drift arrives - with no human
trigger.
"""

import tempfile

from aegis.control_plane.detectors import DriftSeverity
from aegis.system import build_system, run_stream


def test_healthy_stream_opens_no_incidents():
    s = build_system(workdir=tempfile.mkdtemp(prefix="aegis_mon_"))
    summary = run_stream(s, drifted=False)
    assert summary["incidents_opened"] == 0        # zero false interventions
    assert summary["final_state"] == "healthy"


def test_drift_auto_opens_and_resolves_incident():
    s = build_system(workdir=tempfile.mkdtemp(prefix="aegis_mon_"))
    summary = run_stream(s, drifted=True)
    assert summary["incidents_opened"] == 1        # opened without a human
    assert "medium" in summary["detections"]
    assert summary["final_state"] == "healthy"     # loop closed autonomously
    # an incident was persisted with a full audit trail
    inc_id = s.controller.incident_history[-1].incident_id
    assert len(s.incident_store.audit_trail(inc_id)) >= 7


def test_single_feature_blip_is_below_the_incident_bar():
    s = build_system(workdir=tempfile.mkdtemp(prefix="aegis_mon_"))
    batch = s.healthy_stream.iloc[:300].copy()
    batch["f0"] = batch["f0"] + 3.0               # shift exactly one feature
    result = s.monitor.process_batch(batch)
    assert result.severity in (DriftSeverity.LOW, DriftSeverity.NONE)
    assert s.monitor.incidents_opened == 0        # detected, but not incident-worthy
