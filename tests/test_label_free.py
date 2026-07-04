"""
CBPE / DLE label-free estimation - tested on properties that must hold for a
*correct* estimator, not on magic numbers.
"""

import numpy as np
import pandas as pd

from aegis.validation.label_free import LabelFreeValidator


def test_all_uncertain_predictions_estimate_half_accuracy():
    # p = 0.5 everywhere -> a coin flip -> estimated accuracy 0.5
    v = LabelFreeValidator()
    r = v.estimate_cbpe(None, None, np.full(1000, 0.5))
    assert abs(r.estimated_performance - 0.5) < 1e-6
    assert r.confidence == "low"


def test_perfectly_separated_predictions_estimate_full_accuracy():
    # Calibrated, confident scores split cleanly around the threshold.
    proba = np.concatenate([np.full(500, 0.99), np.full(500, 0.01)])
    v = LabelFreeValidator()
    r = v.estimate_cbpe(None, None, proba)
    assert r.estimated_performance > 0.98
    assert r.confidence == "high"


def test_estimate_matches_true_accuracy_on_calibrated_data():
    # Ground-truth check: draw labels from the very probabilities we feed in,
    # so the scores are calibrated by construction. CBPE (no labels) should
    # land close to the accuracy we could only compute *with* labels.
    rng = np.random.RandomState(0)
    proba = rng.uniform(0, 1, 20000)
    labels = (rng.uniform(0, 1, 20000) < proba).astype(int)
    true_acc = ((proba >= 0.5).astype(int) == labels).mean()

    est = LabelFreeValidator().estimate_cbpe(None, None, proba).estimated_performance
    assert abs(est - true_acc) < 0.02  # within 2 points, no labels used


def test_regressive_challenger_is_blocked_by_the_gate():
    # Challenger is barely better than a coin flip; champion baseline is strong.
    weak = np.random.RandomState(1).uniform(0.45, 0.55, 500)
    out = LabelFreeValidator().validate(weak, baseline_performance=0.90)
    assert out["passed"] is False
    assert out["improvement"] < 0


def test_strong_challenger_passes_the_gate():
    strong = np.concatenate([np.full(400, 0.95), np.full(400, 0.05)])
    out = LabelFreeValidator().validate(strong, baseline_performance=0.70)
    assert out["passed"] is True
    assert out["improvement"] > 0


def test_baseline_derived_from_reference_labels():
    # Reference frame carries champion scores + real labels -> measured baseline.
    ref = pd.DataFrame({"proba": [0.9, 0.8, 0.2, 0.1], "is_fraud": [1, 1, 0, 0]})
    r = LabelFreeValidator().estimate_cbpe(ref, None, np.full(100, 0.99))
    assert r.baseline_performance == 1.0  # champion was perfect on reference


def test_empty_predictions_do_not_crash():
    r = LabelFreeValidator().estimate_cbpe(None, None, np.array([]))
    assert r.passed is False
    assert r.confidence == "none"
