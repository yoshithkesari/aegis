"""
Controller - Deterministic state machine for incident lifecycle

Outer loop: deterministic controller that drives the incident state machine
- HEALTHY → DRIFT_SUSPECTED → INVESTIGATING → DIAGNOSED → RETRAINING → VALIDATING → CANARY → PROMOTED
- Escalation paths: ESCALATED (human), ROLLED_BACK
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import logging


class IncidentState(Enum):
    """States in the incident lifecycle"""
    HEALTHY = "healthy"
    DRIFT_SUSPECTED = "drift_suspected"
    INVESTIGATING = "investigating"
    DIAGNOSED = "diagnosed"
    RETRAINING = "retraining"
    VALIDATING = "validating"
    CANARY = "canary"
    PROMOTED = "promoted"
    ESCALATED = "escalated"
    ROLLED_BACK = "rolled_back"


@dataclass
class Incident:
    """Represents an incident"""
    incident_id: str
    model_id: str
    state: IncidentState
    created_at: str
    updated_at: str
    drift_type: str
    severity: str
    diagnosis: Optional[str] = None
    root_cause: Optional[str] = None
    recommended_action: Optional[str] = None
    retrain_job_id: Optional[str] = None
    validation_result: Optional[Dict[str, Any]] = None
    canary_metrics: Optional[Dict[str, Any]] = None
    # challenger evidence the retrain step attaches for the label-free gate:
    # {"challenger_proba": array, "baseline": float, "reference": DataFrame}
    validation_context: Optional[Dict[str, Any]] = None


class Controller:
    """
    Deterministic controller for incident state machine
    
    The controller is the safety-critical path - it must be:
    - Predictable
    - Idempotent
    - Auditable
    - LLM-free (deterministic only)
    """
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.current_state = IncidentState.HEALTHY
        self.current_incident: Optional[Incident] = None
        self.incident_history: list = []
        self.logger = logging.getLogger(f"Controller.{model_id}")
        
        # Dependencies (injected)
        self.detectors = None
        self.risk_gate = None
        self.remediation = None
        self.investigator = None
        self.validator = None
    
    def set_dependencies(
        self,
        detectors,
        risk_gate,
        remediation,
        investigator,
        validator
    ):
        """Inject dependencies"""
        self.detectors = detectors
        self.risk_gate = risk_gate
        self.remediation = remediation
        self.investigator = investigator
        self.validator = validator
    
    def transition_to(self, new_state: IncidentState, reason: str = ""):
        """Transition to a new state"""
        old_state = self.current_state
        self.current_state = new_state
        
        self.logger.info(f"State transition: {old_state.value} → {new_state.value} ({reason})")
        
        if self.current_incident:
            self.current_incident.state = new_state
            self.current_incident.updated_at = datetime.utcnow().isoformat()
    
    def handle_drift_detected(self, drift_result: Dict[str, Any]) -> Incident:
        """Handle drift detection - open incident"""
        if self.current_state != IncidentState.HEALTHY:
            self.logger.warning(f"Ignoring drift detection - not in HEALTHY state (current: {self.current_state.value})")
            return self.current_incident
        
        # Create new incident
        incident = Incident(
            incident_id=f"inc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            model_id=self.model_id,
            state=IncidentState.DRIFT_SUSPECTED,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            drift_type=drift_result.get("drift_type", "unknown"),
            severity=drift_result.get("severity", "unknown")
        )
        
        self.current_incident = incident
        self.incident_history.append(incident)
        self.transition_to(IncidentState.DRIFT_SUSPECTED, f"Drift detected: {drift_result.get('summary', '')}")
        
        # Move to investigation
        self.start_investigation()
        
        return incident
    
    def start_investigation(self):
        """Start agentic investigation (inner loop)"""
        if self.current_state != IncidentState.DRIFT_SUSPECTED:
            return
        
        self.transition_to(IncidentState.INVESTIGATING, "Starting root cause investigation")
        
        # Call investigator (LLM-powered, read-only)
        if self.investigator:
            diagnosis = self.investigator.investigate(self.current_incident)

            self.current_incident.diagnosis = diagnosis.diagnosis
            self.current_incident.root_cause = diagnosis.root_cause
            self.current_incident.recommended_action = diagnosis.recommended_action

            self.transition_to(IncidentState.DIAGNOSED, f"Investigation complete: {diagnosis.root_cause or 'unknown'}")
            
            # Move to risk gate decision
            self.evaluate_risk_gate()
    
    def evaluate_risk_gate(self):
        """Evaluate risk gate for autonomous decision"""
        if self.current_state != IncidentState.DIAGNOSED:
            return
        
        if not self.risk_gate:
            self.transition_to(IncidentState.ESCALATED, "Risk gate not available - escalating")
            return
        
        # Get decision from risk gate
        decision = self.risk_gate.evaluate(
            drift_severity=self.current_incident.severity,
            labels_available=True,  # TODO: determine from context
            label_free_estimate=None,  # TODO: get from validator
            current_performance=None,
            baseline_performance=None
        )
        
        self.logger.info(f"Risk gate decision: {decision.action.value} - {decision.reason}")
        
        if decision.action.value == "escalate":
            self.transition_to(IncidentState.ESCALATED, f"Risk gate escalation: {decision.reason}")
        elif decision.action.value == "block":
            self.transition_to(IncidentState.ROLLED_BACK, f"Risk gate block: {decision.reason}")
        elif "retrain" in decision.action.value:
            self.start_retraining(decision.action.value == "auto_retrain_shadow_hold")
        else:
            self.transition_to(IncidentState.HEALTHY, f"Risk gate: {decision.action.value}")
    
    def start_retraining(self, shadow_hold: bool = False):
        """Start retraining process"""
        if self.current_state != IncidentState.DIAGNOSED:
            return
        
        self.transition_to(IncidentState.RETRAINING, "Starting model retraining")
        
        # Call remediation to retrain
        if self.remediation:
            retrain_result = self.remediation.retrain(self.current_incident)
            
            self.current_incident.retrain_job_id = retrain_result.get("job_id")
            
            if retrain_result.get("success"):
                # start_validation guards on RETRAINING and performs the
                # transition itself - don't pre-transition here or it no-ops.
                self.start_validation(shadow_hold)
            else:
                self.transition_to(IncidentState.ESCALATED, f"Retraining failed: {retrain_result.get('error')}")
    
    def start_validation(self, shadow_hold: bool = False):
        """Start validation (label-free if needed)"""
        if self.current_state != IncidentState.RETRAINING:
            return
        
        self.transition_to(IncidentState.VALIDATING, "Starting validation")
        
        # Call validator
        if self.validator:
            validation_result = self.validator.validate(
                self.current_incident,
                label_free=shadow_hold
            )
            
            self.current_incident.validation_result = validation_result
            
            if validation_result.get("passed"):
                if shadow_hold:
                    self.transition_to(IncidentState.HEALTHY, "Validation passed - shadow hold until labels confirm")
                else:
                    self.start_canary()
            else:
                self.transition_to(IncidentState.ROLLED_BACK, f"Validation failed: {validation_result.get('reason')}")
    
    def start_canary(self):
        """Start canary deployment"""
        if self.current_state != IncidentState.VALIDATING:
            return
        
        self.transition_to(IncidentState.CANARY, "Starting canary deployment")
        
        # Call remediation to deploy canary
        if self.remediation:
            canary_result = self.remediation.deploy_canary(self.current_incident)
            
            self.current_incident.canary_metrics = canary_result.get("metrics")
            
            if canary_result.get("success"):
                # promote_challenger guards on CANARY and performs the
                # PROMOTED transition itself - don't pre-transition here.
                self.promote_challenger()
            else:
                self.transition_to(IncidentState.ROLLED_BACK, f"Canary failed: {canary_result.get('error')}")
    
    def promote_challenger(self):
        """Promote challenger to champion"""
        if self.current_state != IncidentState.CANARY:
            return
        
        # Call remediation to promote
        if self.remediation:
            promote_result = self.remediation.promote(self.current_incident)
            
            if promote_result.get("success"):
                self.transition_to(IncidentState.PROMOTED, "Challenger promoted")
                self.transition_to(IncidentState.HEALTHY, "Incident resolved")
                self.current_incident = None
            else:
                self.transition_to(IncidentState.ROLLED_BACK, f"Promotion failed: {promote_result.get('error')}")
    
    def escalate_to_human(self, reason: str):
        """Escalate incident to human"""
        self.transition_to(IncidentState.ESCALATED, f"Escalating to human: {reason}")
        
        # In production, would send alert via Slack/PagerDuty
        self.logger.error(f"INCIDENT ESCALATED: {reason}")
        self.logger.error(f"Incident ID: {self.current_incident.incident_id if self.current_incident else 'N/A'}")
    
    def rollback(self, reason: str):
        """Rollback to previous model"""
        self.transition_to(IncidentState.ROLLED_BACK, f"Rolling back: {reason}")
        
        if self.remediation:
            rollback_result = self.remediation.rollback(self.current_incident)
            
            if rollback_result.get("success"):
                self.transition_to(IncidentState.HEALTHY, "Rollback successful")
                self.current_incident = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current controller status"""
        return {
            "model_id": self.model_id,
            "current_state": self.current_state.value,
            "current_incident": self.current_incident.__dict__ if self.current_incident else None,
            "incident_count": len(self.incident_history)
        }
