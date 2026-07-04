"""
Risk Gate - Decision matrix for autonomous remediation

Determines when to auto-fix vs escalate based on:
- Drift severity
- Label availability
- Label-free recovery estimate
- Would regression occur
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class Action(Enum):
    """Actions the controller can take"""
    LOG_ONLY = "log_only"
    AUTO_RETRAIN = "auto_retrain"
    AUTO_RETRAIN_SHADOW_HOLD = "auto_retrain_shadow_hold"
    ESCALATE = "escalate"
    BLOCK = "block"


@dataclass
class RiskGateDecision:
    """Decision from the risk gate"""
    action: Action
    reason: str
    confidence: float
    requires_human: bool
    rollback_armed: bool


class RiskGate:
    """
    Risk-gated autonomy decision matrix
    
    Drift severity | Labels? | Label-free recovery est. | Controller action
    ---------------|---------|------------------------|-------------------
    Low            | —       | —                      | Log only
    Medium         | Yes     | —                      | Auto-retrain → validate → canary → promote
    Medium         | Delayed | High                   | Auto-retrain → shadow-hold until labels confirm
    High           | any     | any                    | Escalate + recommended fix + rollback armed
    any            | —       | Would regress          | Block — keep champion, alert
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.05,
        performance_threshold: float = 0.1,
        regression_threshold: float = -0.05
    ):
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.regression_threshold = regression_threshold
    
    def evaluate(
        self,
        drift_severity: str,
        labels_available: bool,
        label_free_estimate: Optional[float] = None,
        current_performance: Optional[float] = None,
        baseline_performance: Optional[float] = None
    ) -> RiskGateDecision:
        """
        Evaluate whether to auto-remediate or escalate
        
        Args:
            drift_severity: 'none', 'low', 'medium', 'high', 'critical'
            labels_available: Whether ground-truth labels are available
            label_free_estimate: Estimated performance of retrained model (if available)
            current_performance: Current model performance
            baseline_performance: Baseline (champion) performance
        
        Returns:
            RiskGateDecision with action and reasoning
        """
        # Check for regression
        would_regress = False
        if (label_free_estimate is not None and 
            baseline_performance is not None and
            label_free_estimate < baseline_performance + self.regression_threshold):
            would_regress = True
        
        if would_regress:
            return RiskGateDecision(
                action=Action.BLOCK,
                reason="Label-free estimate indicates retrain would regress performance",
                confidence=0.9,
                requires_human=False,
                rollback_armed=False
            )
        
        # High severity - always escalate
        if drift_severity in ["high", "critical"]:
            return RiskGateDecision(
                action=Action.ESCALATE,
                reason=f"High severity drift ({drift_severity}) requires human intervention",
                confidence=0.95,
                requires_human=True,
                rollback_armed=True
            )
        
        # Low severity - log only
        if drift_severity == "low":
            return RiskGateDecision(
                action=Action.LOG_ONLY,
                reason="Low severity drift - logging for trend analysis",
                confidence=0.8,
                requires_human=False,
                rollback_armed=False
            )
        
        # Medium severity
        if drift_severity == "medium":
            if labels_available:
                # Labels available - safe to auto-retrain with full validation
                return RiskGateDecision(
                    action=Action.AUTO_RETRAIN,
                    reason="Medium severity drift with labels available - safe to auto-retrain",
                    confidence=0.85,
                    requires_human=False,
                    rollback_armed=True
                )
            else:
                # Labels delayed - check label-free estimate
                if label_free_estimate and label_free_estimate > baseline_performance + 0.05:
                    # Label-free estimate shows improvement - shadow hold
                    return RiskGateDecision(
                        action=Action.AUTO_RETRAIN_SHADOW_HOLD,
                        reason="Medium severity drift, labels delayed but label-free estimate shows improvement - shadow deploy until labels confirm",
                        confidence=0.75,
                        requires_human=False,
                        rollback_armed=True
                    )
                else:
                    # No label-free estimate or uncertain escalate
                    return RiskGateDecision(
                        action=Action.ESCALATE,
                        reason="Medium severity drift without labels and uncertain label-free estimate - escalate for human review",
                        confidence=0.7,
                        requires_human=True,
                        rollback_armed=True
                    )
        
        # Default - log only
        return RiskGateDecision(
            action=Action.LOG_ONLY,
            reason="Default action - log only",
            confidence=0.5,
            requires_human=False,
            rollback_armed=False
        )
    
    def can_auto_retrain(self, decision: RiskGateDecision) -> bool:
        """Check if decision allows auto-retraining"""
        return decision.action in [Action.AUTO_RETRAIN, Action.AUTO_RETRAIN_SHADOW_HOLD]
    
    def requires_escalation(self, decision: RiskGateDecision) -> bool:
        """Check if decision requires human escalation"""
        return decision.requires_human or decision.action == Action.ESCALATE
