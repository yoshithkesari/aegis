"""
Stream monitor - closes the loop from a live prediction stream to an incident.

This is the missing edge: batches arrive, the detector compares them to the
reference distribution, and when drift crosses the threshold the monitor opens
an incident on the controller *automatically* (no human trigger). The controller
then runs its deterministic remediation loop as usual.

The monitor holds no write authority - it only reads batches and calls
handle_drift_detected, which is the controller's single entry point.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .controller import IncidentState
from .detectors import DetectorSuite, DriftDetectionResult, DriftSeverity

# Ordering so a minimum-severity gate can compare severities.
_SEV_RANK = {
    DriftSeverity.NONE: 0,
    DriftSeverity.LOW: 1,
    DriftSeverity.MEDIUM: 2,
    DriftSeverity.HIGH: 3,
    DriftSeverity.CRITICAL: 4,
}


class StreamMonitor:
    """Watch a batch stream; auto-open an incident when drift is detected.

    Two guards keep this from firing on noise:
    * a **Bonferroni-corrected** per-feature threshold (0.05 / n_features), so
      testing many features at once doesn't manufacture false positives; and
    * a **minimum severity** gate, so a single incidental feature blip is logged
      by the detector but does not open an incident.
    """

    def __init__(
        self,
        controller,
        reference_features: pd.DataFrame,
        detector: Optional[DetectorSuite] = None,
        threshold: Optional[float] = None,
        min_severity: DriftSeverity = DriftSeverity.MEDIUM,
        use_graph: bool = False,
    ):
        self.controller = controller
        self.reference = reference_features
        self.detector = detector or DetectorSuite()
        n_features = max(1, reference_features.shape[1])
        # Bonferroni correction across the features we test each batch.
        self.threshold = threshold if threshold is not None else 0.05 / n_features
        self.min_severity = min_severity
        # When True, incidents are sequenced by the LangGraph orchestrator
        # instead of the controller's own autopilot chaining.
        self.use_graph = use_graph
        self.batches_seen = 0
        self.incidents_opened = 0

    def _open_incident(self, drift: dict) -> None:
        if self.use_graph:
            from ..orchestration import run_incident_via_graph
            run_incident_via_graph(self.controller, drift)
        else:
            self.controller.handle_drift_detected(drift)

    def process_batch(self, batch: pd.DataFrame) -> Optional[DriftDetectionResult]:
        """Run drift detection on one batch; open an incident if warranted."""
        self.batches_seen += 1
        cols = [c for c in self.reference.columns if c in batch.columns]
        result = self.detector.detect_batch_drift(
            self.reference[cols], batch[cols], threshold=self.threshold
        )

        severe_enough = _SEV_RANK[result.severity] >= _SEV_RANK[self.min_severity]
        # Only open from the steady state - the controller ignores drift while an
        # incident is already in flight, so this stays idempotent.
        if (result.drift_detected and severe_enough
                and self.controller.current_state is IncidentState.HEALTHY):
            self.incidents_opened += 1
            self._open_incident({
                "drift_type": result.drift_type,
                "severity": result.severity.value,
                "summary": result.summary,
                "affected_features": result.affected_features,
            })
            # If the controller remediated and returned to steady state, the
            # remediated window is the new champion's normal - adopt it as the
            # reference so we don't re-open the same drift every batch.
            if self.controller.current_state is IncidentState.HEALTHY:
                self.reference = batch[cols].copy()
        return result

    def run(self, replayer) -> Dict[str, Any]:
        """Drive an entire replayer stream through the monitor."""
        detections: List[str] = []
        for batch in replayer.replay():
            result = self.process_batch(batch)
            if result and result.drift_detected:
                detections.append(result.severity.value)
        return {
            "batches_seen": self.batches_seen,
            "incidents_opened": self.incidents_opened,
            "detections": detections,
            "final_state": self.controller.current_state.value,
        }
