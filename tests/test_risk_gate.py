"""The risk gate is pure decision logic - the safety-critical core. Test it hard."""

from aegis.control_plane.risk_gate import Action, RiskGate


def gate():
    return RiskGate()


def test_low_severity_logs_only():
    d = gate().evaluate(drift_severity="low", labels_available=False)
    assert d.action is Action.LOG_ONLY
    assert d.requires_human is False


def test_high_severity_always_escalates_with_rollback_armed():
    for labels in (True, False):
        d = gate().evaluate(drift_severity="high", labels_available=labels)
        assert d.action is Action.ESCALATE
        assert d.requires_human is True
        assert d.rollback_armed is True


def test_critical_is_treated_as_high():
    d = gate().evaluate(drift_severity="critical", labels_available=True)
    assert d.action is Action.ESCALATE


def test_medium_with_labels_auto_retrains():
    d = gate().evaluate(drift_severity="medium", labels_available=True)
    assert d.action is Action.AUTO_RETRAIN
    assert d.rollback_armed is True


def test_medium_delayed_labels_but_strong_estimate_shadow_holds():
    d = gate().evaluate(
        drift_severity="medium",
        labels_available=False,
        label_free_estimate=0.90,
        baseline_performance=0.80,
    )
    assert d.action is Action.AUTO_RETRAIN_SHADOW_HOLD


def test_medium_delayed_labels_weak_estimate_escalates():
    d = gate().evaluate(
        drift_severity="medium",
        labels_available=False,
        label_free_estimate=0.81,
        baseline_performance=0.80,
    )
    assert d.action is Action.ESCALATE
    assert d.requires_human is True


def test_regressive_retrain_is_blocked():
    # Estimated performance well below champion -> never ship it.
    d = gate().evaluate(
        drift_severity="medium",
        labels_available=True,
        label_free_estimate=0.60,
        baseline_performance=0.90,
    )
    assert d.action is Action.BLOCK


def test_can_auto_retrain_helper():
    g = gate()
    auto = g.evaluate(drift_severity="medium", labels_available=True)
    esc = g.evaluate(drift_severity="high", labels_available=True)
    assert g.can_auto_retrain(auto) is True
    assert g.can_auto_retrain(esc) is False
    assert g.requires_escalation(esc) is True
