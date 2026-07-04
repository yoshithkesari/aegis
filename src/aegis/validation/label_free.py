"""
Label-free validation - a real Confidence-Based Performance Estimate (CBPE).

This is the "hero technique": it lets the deploy gate estimate a challenger's
live performance *without ground truth*, so the loop can close at decision time
instead of weeks later when labels arrive.

How it actually works (no magic, no hardcoded baseline):

    For a binary classifier that emits a *calibrated* probability p_i = P(y=1|x_i)
    and a decision threshold t, the predicted label is  y_hat_i = 1[p_i >= t].
    Under calibration, the expected confusion-matrix contributions of sample i are

        y_hat=1:  E[TP] += p_i        E[FP] += (1 - p_i)
        y_hat=0:  E[TN] += (1 - p_i)  E[FN] += p_i

    Summing over the unlabeled current window yields an expected confusion matrix,
    and every rate metric (accuracy, precision, recall, F1) follows from it. This
    is exactly what NannyML's CBPE does; we implement the estimator directly so it
    runs anywhere (incl. Python >=3.13, where NannyML has no wheel).

Calibration matters, so when reference predictions + labels are available we fit
an isotonic calibrator on them first; otherwise we assume the scores are already
calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# Flexible column detection so callers don't have to rename things.
_PROBA_COLS = ("proba", "prediction_proba", "y_proba", "score", "pred_proba")
_LABEL_COLS = ("is_fraud", "target", "label", "y", "y_true")


@dataclass
class ValidationResult:
    """Result of label-free validation."""
    passed: bool
    estimated_performance: float
    baseline_performance: float
    improvement: float
    confidence: str
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)


def _as_proba(values: Sequence[float]) -> np.ndarray:
    """Coerce to a 1-D float array of probabilities in [0, 1]."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return arr
    # If it looks like logits / raw scores outside [0,1], squash with a sigmoid.
    if arr.min() < 0.0 or arr.max() > 1.0:
        arr = 1.0 / (1.0 + np.exp(-arr))
    return np.clip(arr, 0.0, 1.0)


def _extract(df: Optional[pd.DataFrame]):
    """Pull (proba, labels) out of a frame using flexible column names."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, None
    proba = next((df[c].to_numpy() for c in _PROBA_COLS if c in df.columns), None)
    labels = next((df[c].to_numpy() for c in _LABEL_COLS if c in df.columns), None)
    return (_as_proba(proba) if proba is not None else None,
            None if labels is None else np.asarray(labels, dtype=float).ravel())


def _expected_confusion(proba: np.ndarray, threshold: float):
    """Expected (TP, FP, TN, FN) on unlabeled data given calibrated probs."""
    pred_pos = proba >= threshold
    tp = float(proba[pred_pos].sum())
    fp = float((1.0 - proba[pred_pos]).sum())
    tn = float((1.0 - proba[~pred_pos]).sum())
    fn = float(proba[~pred_pos].sum())
    return tp, fp, tn, fn


def _rates(tp: float, fp: float, tn: float, fn: float) -> Dict[str, float]:
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class LabelFreeValidator:
    """Estimate a retrain's live performance without ground-truth labels.

    Parameters
    ----------
    threshold : decision threshold used to derive predicted labels.
    pass_margin : minimum estimated improvement over baseline to pass the gate.
    """

    def __init__(self, threshold: float = 0.5, pass_margin: float = 0.02):
        self.threshold = threshold
        self.pass_margin = pass_margin
        self._calibrator = None  # optional isotonic map fitted on reference

    # ---- calibration -----------------------------------------------------
    def fit_calibration(self, ref_proba: Sequence[float], ref_labels: Sequence[float]) -> "LabelFreeValidator":
        """Fit an isotonic calibrator on reference (proba, label) pairs.

        CBPE assumes calibrated scores; this makes that assumption hold as
        closely as the reference data allows. Silently no-ops if sklearn or
        the data isn't available.
        """
        p = _as_proba(ref_proba)
        y = np.asarray(ref_labels, dtype=float).ravel()
        if p.size < 10 or p.size != y.size or len(np.unique(y)) < 2:
            return self
        try:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y)
            self._calibrator = iso
        except Exception:
            self._calibrator = None
        return self

    def _calibrate(self, proba: np.ndarray) -> np.ndarray:
        if self._calibrator is None:
            return proba
        return np.clip(self._calibrator.predict(proba), 0.0, 1.0)

    # ---- baseline --------------------------------------------------------
    def _baseline_from_reference(self, reference_data: Optional[pd.DataFrame]) -> Optional[float]:
        """Measured champion accuracy on reference (uses real labels if present)."""
        ref_proba, ref_labels = _extract(reference_data)
        if ref_proba is None:
            return None
        if ref_labels is not None and ref_labels.size == ref_proba.size:
            preds = (ref_proba >= self.threshold).astype(float)
            return float((preds == ref_labels).mean())
        # No labels on reference either -> fall back to CBPE on the reference set.
        tp, fp, tn, fn = _expected_confusion(self._calibrate(ref_proba), self.threshold)
        return _rates(tp, fp, tn, fn)["accuracy"]

    # ---- estimators ------------------------------------------------------
    def estimate_cbpe(
        self,
        reference_data: Optional[pd.DataFrame],
        current_data: Optional[pd.DataFrame],
        model_predictions: Sequence[float],
        baseline_performance: Optional[float] = None,
    ) -> ValidationResult:
        """Confidence-Based Performance Estimation on the current window."""
        proba = self._calibrate(_as_proba(model_predictions))
        if proba.size == 0:
            return ValidationResult(False, 0.0, 0.0, 0.0, "none",
                                    "No predictions provided", {})

        # Auto-calibrate from reference if we have labelled reference scores.
        if self._calibrator is None:
            ref_proba, ref_labels = _extract(reference_data)
            if ref_proba is not None and ref_labels is not None:
                self.fit_calibration(ref_proba, ref_labels)
                proba = self._calibrate(_as_proba(model_predictions))

        tp, fp, tn, fn = _expected_confusion(proba, self.threshold)
        rates = _rates(tp, fp, tn, fn)
        estimated = rates["accuracy"]

        if baseline_performance is None:
            baseline_performance = self._baseline_from_reference(reference_data)
        if baseline_performance is None:
            baseline_performance = estimated  # neutral: no evidence of change

        improvement = estimated - baseline_performance
        # Confidence: sharper predictions (mass near 0/1) => more trustworthy est.
        sharpness = float(np.mean(np.abs(proba - 0.5)) * 2.0)  # 0 (all 0.5) .. 1
        confidence = "high" if sharpness > 0.6 else "medium" if sharpness > 0.3 else "low"

        return ValidationResult(
            passed=improvement >= self.pass_margin,
            estimated_performance=estimated,
            baseline_performance=baseline_performance,
            improvement=improvement,
            confidence=confidence,
            reason=(f"CBPE estimated accuracy {estimated:.3f} vs baseline "
                    f"{baseline_performance:.3f} (Δ{improvement:+.3f}), "
                    f"sharpness {sharpness:.2f}"),
            metrics={
                "estimated_accuracy": estimated,
                "estimated_precision": rates["precision"],
                "estimated_recall": rates["recall"],
                "estimated_f1": rates["f1"],
                "baseline_accuracy": baseline_performance,
                "improvement": improvement,
                "sharpness": sharpness,
                "n": float(proba.size),
            },
        )

    def estimate_dle(
        self,
        reference_data: Optional[pd.DataFrame],
        current_data: Optional[pd.DataFrame],
        model_predictions: Sequence[float],
        baseline_performance: Optional[float] = None,
    ) -> ValidationResult:
        """Direct Loss Estimation: estimate expected 0-1 loss without labels.

        Expected loss for sample i is the probability of being wrong:
            y_hat=1 -> (1 - p_i);  y_hat=0 -> p_i
        Estimated accuracy = 1 - mean(expected loss). (Equivalent to CBPE
        accuracy here, but framed via loss and reported independently so the
        combined estimate can cross-check the two.)
        """
        proba = self._calibrate(_as_proba(model_predictions))
        if proba.size == 0:
            return ValidationResult(False, 0.0, 0.0, 0.0, "none",
                                    "No predictions provided", {})
        pred_pos = proba >= self.threshold
        expected_loss = np.where(pred_pos, 1.0 - proba, proba)
        estimated = float(1.0 - expected_loss.mean())

        if baseline_performance is None:
            baseline_performance = self._baseline_from_reference(reference_data)
        if baseline_performance is None:
            baseline_performance = estimated
        improvement = estimated - baseline_performance

        return ValidationResult(
            passed=improvement >= self.pass_margin,
            estimated_performance=estimated,
            baseline_performance=baseline_performance,
            improvement=improvement,
            confidence="medium",
            reason=f"DLE estimated accuracy {estimated:.3f} (mean expected loss "
                   f"{expected_loss.mean():.3f})",
            metrics={
                "estimated_accuracy": estimated,
                "mean_expected_loss": float(expected_loss.mean()),
                "baseline_accuracy": baseline_performance,
                "improvement": improvement,
            },
        )

    def combined_estimate(
        self,
        reference_data: Optional[pd.DataFrame],
        current_data: Optional[pd.DataFrame],
        model_predictions: Sequence[float],
        baseline_performance: Optional[float] = None,
    ) -> ValidationResult:
        """Average CBPE and DLE; pass only if they agree the gate is safe.

        Requiring both methods to clear the margin is the conservative choice -
        it makes a regressive retrain harder to slip through the deploy gate.
        """
        cbpe = self.estimate_cbpe(reference_data, current_data, model_predictions, baseline_performance)
        dle = self.estimate_dle(reference_data, current_data, model_predictions, baseline_performance)
        if cbpe.metrics.get("n", 0) == 0:
            return cbpe

        combined = (cbpe.estimated_performance + dle.estimated_performance) / 2.0
        baseline = cbpe.baseline_performance
        improvement = combined - baseline
        passed = cbpe.passed and dle.passed  # both must agree

        return ValidationResult(
            passed=passed,
            estimated_performance=combined,
            baseline_performance=baseline,
            improvement=improvement,
            confidence="high" if cbpe.confidence == "high" else "medium",
            reason=f"CBPE+DLE combined accuracy {combined:.3f} vs baseline "
                   f"{baseline:.3f} (both-agree gate: {'PASS' if passed else 'HOLD'})",
            metrics={
                "cbpe_estimate": cbpe.estimated_performance,
                "dle_estimate": dle.estimated_performance,
                "combined_estimate": combined,
                "baseline_accuracy": baseline,
                "improvement": improvement,
            },
        )

    def validate(self, challenger_proba, baseline_performance=None, reference_data=None):
        """Controller-facing gate. Returns a dict the state machine can act on."""
        result = self.combined_estimate(reference_data, None, challenger_proba, baseline_performance)
        return {
            "passed": result.passed,
            "reason": result.reason,
            "estimated_performance": result.estimated_performance,
            "baseline_performance": result.baseline_performance,
            "improvement": result.improvement,
            "confidence": result.confidence,
            "metrics": result.metrics,
        }
