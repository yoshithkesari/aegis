"""
Composition-root test: the whole system, wired with real components, closes the
loop and leaves a durable audit trail.
"""

import tempfile

from aegis.system import build_system, demo_run, run_incident


def test_demo_run_produces_real_recovery_payload():
    d = demo_run()
    # healthy stream is quiet; drift auto-opens exactly one incident that resolves
    assert d["healthy_incidents"] == 0
    assert d["drift_incidents"] == 1
    assert d["final_state"] == "healthy"
    # label-free gate promoted, and the challenger really did recover accuracy
    assert d["gate"]["passed"] is True
    assert d["challenger_acc"] >= d["champion_acc"]
    assert d["severity"] == "medium"
    # dramatic-but-real before/after: drift degrades the champion, AEGIS recovers
    assert d["champion_healthy_acc"] > d["champion_drift_acc"] + 0.15
    assert d["recovery"] > 0.15
    assert len(d["audit_trail"]) >= 7


def test_full_incident_closes_the_loop_and_persists_audit_trail():
    system = build_system(workdir=tempfile.mkdtemp(prefix="aegis_test_"))
    result = run_incident(system, severity="medium")

    # loop closed
    assert result["final_state"] == "healthy"

    # real label-free gate ran and cleared the margin
    assert result["gate"]["passed"] is True
    assert result["gate"]["estimated_performance"] > result["gate"]["baseline_performance"]

    # registry advanced past the initial champion (v1 champion -> promoted challenger)
    assert result["champion_version"] is not None

    # durable audit trail captured the whole lifecycle
    states = [(e["from_state"], e["to_state"]) for e in result["audit_trail"]]
    assert ("healthy", "drift_suspected") in states
    assert ("canary", "promoted") in states
    assert len(states) >= 7

    # incident is retrievable from the store after resolution
    stored = system.incident_store.get(result["incident_id"])
    assert stored is not None
    assert stored["state"] == "healthy"


def test_low_severity_incident_only_logs_no_promotion():
    system = build_system(workdir=tempfile.mkdtemp(prefix="aegis_test_"))
    champ_before = system.registry.get_champion().version
    run_incident(system, severity="low")
    # low severity -> log only -> controller returns to healthy, no new champion
    assert system.registry.get_champion().version == champ_before
