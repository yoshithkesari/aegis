"""
Investigator - Agentic inner loop for root cause investigation.

The LLM is scoped to exactly one job: root-cause investigation. It can call
read-only tools but cannot deploy. The reasoning call itself sits behind the
pluggable `Reasoner` interface, so the hosted free-tier model, a local model,
or a no-LLM heuristic are all one-line swaps.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .tools import InvestigationToolkit
from ..reasoning import Reasoner, ReasonerError, build_reasoner


@dataclass
class InvestigationResult:
    """Result of agentic investigation"""
    diagnosis: str
    root_cause: str
    recommended_action: str
    confidence: float
    tool_calls: List[Dict[str, Any]]
    reasoning: str


class Investigator:
    """
    Agentic investigator using an LLM for root cause analysis.

    The investigator follows a loop:
    hypothesize -> call diagnostic tool -> observe -> refine -> conclude

    The investigator is read-only - it can investigate but cannot deploy. All
    reasoning goes through a `Reasoner`, which is read-only by contract.
    """

    def __init__(self, reasoner: Optional[Reasoner] = None):
        # Defaults to the hosted->local->heuristic fallback chain.
        self.reasoner = reasoner or build_reasoner()

    def investigate(self, incident) -> InvestigationResult:
        """
        Conduct root cause investigation.

        Args:
            incident: The incident object with drift information.

        Returns:
            InvestigationResult with diagnosis and recommendations.
        """
        context = self._build_context(incident)
        tool_results = self._run_tools(incident)

        prompt = self._build_prompt(context, tool_results)
        try:
            text = self.reasoner.complete(prompt)
            return self._parse_response(text, tool_results)
        except ReasonerError:
            # Every backend exhausted (offline, capped) - deterministic fallback.
            return self._rule_based_fallback_from_tools(context, tool_results)

    def _build_context(self, incident) -> Dict[str, Any]:
        """Build investigation context from incident"""
        return {
            "incident_id": incident.incident_id,
            "model_id": incident.model_id,
            "drift_type": incident.drift_type,
            "severity": incident.severity,
            "created_at": incident.created_at,
        }

    def _run_tools(self, incident) -> Dict[str, Any]:
        """Run investigation tools (read-only).

        In production these read real data from the incident; for the demo we
        return representative results so the loop is exercisable end to end.
        """
        return {
            "slice_performance": {
                "summary": "Performance degradation observed in 'online' merchant category",
                "details": [
                    {"feature": "merchant_category", "value": "online", "accuracy": 0.67, "count": 500}
                ],
            },
            "attribute_drift": {
                "summary": "High drift detected in transaction_amount (PSI: 0.35)",
                "top_drift_features": [
                    {"feature": "transaction_amount", "psi": 0.35, "severity": "high"},
                    {"feature": "merchant_category", "psi": 0.15, "severity": "medium"},
                ],
            },
            "query_deploy_log": {
                "summary": "No recent deployments in last 7 days",
                "recent_deploys": [],
            },
            "diff_schema": {
                "summary": "No schema changes detected",
                "changes": [],
            },
            "estimate_perf_without_labels": {
                "summary": "Estimated accuracy: 0.71 (low confidence)",
                "estimated_accuracy": 0.71,
            },
        }

    def _build_prompt(self, context: Dict[str, Any], tool_results: Dict[str, Any]) -> str:
        """Build prompt for the reasoner (aggregated evidence only - no PII)."""
        return f"""You are an expert ML reliability engineer investigating a model performance incident.

INCIDENT CONTEXT:
- Incident ID: {context['incident_id']}
- Model ID: {context['model_id']}
- Drift Type: {context['drift_type']}
- Severity: {context['severity']}
- Created: {context['created_at']}

INVESTIGATION TOOL RESULTS:

1. Slice Performance Analysis:
{tool_results['slice_performance']['summary']}
Details: {tool_results['slice_performance']['details']}

2. Attribute Drift Analysis:
{tool_results['attribute_drift']['summary']}
Top drifting features: {tool_results['attribute_drift']['top_drift_features']}

3. Deploy Log Query:
{tool_results['query_deploy_log']['summary']}
Recent deployments: {tool_results['query_deploy_log']['recent_deploys']}

4. Schema Diff:
{tool_results['diff_schema']['summary']}
Changes: {tool_results['diff_schema']['changes']}

5. Performance Estimate (without labels):
{tool_results['estimate_perf_without_labels']['summary']}
Estimated: {tool_results['estimate_perf_without_labels']['estimated_accuracy']}

TASK:
Based on the tool results above, provide:
1. DIAGNOSIS: A concise summary of what's happening
2. ROOT CAUSE: The specific cause of the drift/degradation
3. RECOMMENDED ACTION: What should be done to fix it
4. CONFIDENCE: Your confidence in this diagnosis (0-1)

Format your response as:
DIAGNOSIS: [your diagnosis]
ROOT CAUSE: [root cause]
RECOMMENDED ACTION: [recommended action]
CONFIDENCE: [0.0-1.0]

Be specific and technical. Focus on the most likely cause based on the evidence.
"""

    def _parse_response(self, response_text: str, tool_results: Dict[str, Any]) -> InvestigationResult:
        """Parse a reasoner response into a structured result."""
        diagnosis = ""
        root_cause = ""
        recommended_action = ""
        confidence = 0.7

        for line in response_text.strip().split("\n"):
            line = line.strip()
            if line.startswith("DIAGNOSIS:"):
                diagnosis = line.replace("DIAGNOSIS:", "").strip()
            elif line.startswith("ROOT CAUSE:"):
                root_cause = line.replace("ROOT CAUSE:", "").strip()
            elif line.startswith("RECOMMENDED ACTION:"):
                recommended_action = line.replace("RECOMMENDED ACTION:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass

        return InvestigationResult(
            diagnosis=diagnosis or "Performance degradation detected",
            root_cause=root_cause or "Unknown - requires manual investigation",
            recommended_action=recommended_action or "Manual review required",
            confidence=confidence,
            tool_calls=tool_results,
            reasoning=response_text,
        )

    def _rule_based_fallback_from_tools(
        self, context: Dict[str, Any], tool_results: Dict[str, Any]
    ) -> InvestigationResult:
        """Deterministic analysis from tool results when no reasoner is reachable."""
        top_features = tool_results["attribute_drift"]["top_drift_features"]
        top_drift = top_features[0] if top_features else None

        if top_drift and top_drift["severity"] == "high":
            diagnosis = f"High drift detected in {top_drift['feature']}"
            root_cause = f"Distribution shift in {top_drift['feature']} (PSI: {top_drift['psi']:.2f})"
            recommended_action = "Retrain model on recent data"
            confidence = 0.7
        else:
            diagnosis = "Performance degradation detected"
            root_cause = "Multiple features showing moderate drift"
            recommended_action = "Monitor and consider retraining if degradation continues"
            confidence = 0.5

        return InvestigationResult(
            diagnosis=diagnosis,
            root_cause=root_cause,
            recommended_action=recommended_action,
            confidence=confidence,
            tool_calls=tool_results,
            reasoning="Deterministic fallback - no reasoner reachable",
        )
