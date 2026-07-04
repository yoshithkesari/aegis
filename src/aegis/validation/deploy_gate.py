"""
Deploy gate - the controller-facing wrapper around label-free validation.

The controller calls ``validator.validate(incident, label_free=bool)`` and acts
on a plain dict. This adapter reads the challenger evidence that the retrain
step attaches to the incident (``incident.validation_context``) and runs the
CBPE/DLE estimate, so the deterministic controller never touches ML internals.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .invariance_suite import InvarianceSuite
from .label_free import LabelFreeValidator


class DeployGate:
    """Label-free deploy gate. Blocks retrains that don't clear the margin."""

    def __init__(
        self,
        validator: Optional[LabelFreeValidator] = None,
        pass_margin: float = 0.02,
        run_invariance: bool = False,
    ):
        self.validator = validator or LabelFreeValidator(pass_margin=pass_margin)
        self.invariance = InvarianceSuite() if run_invariance else None

    def validate(self, incident, label_free: bool = False) -> Dict[str, Any]:
        ctx: Dict[str, Any] = getattr(incident, "validation_context", None) or {}
        challenger_proba = ctx.get("challenger_proba")

        if challenger_proba is None:
            # No evidence to judge -> fail closed (keep the champion).
            return {
                "passed": False,
                "reason": "No challenger predictions available to validate",
                "estimated_performance": None,
                "label_free": bool(label_free),
            }

        result = self.validator.validate(
            challenger_proba,
            baseline_performance=ctx.get("baseline"),
            reference_data=ctx.get("reference"),
        )
        result["label_free"] = bool(label_free)
        return result
