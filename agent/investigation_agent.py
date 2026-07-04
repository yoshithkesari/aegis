"""
Autonomous Investigation Agent

Uses LLM reasoning to:
- Investigate detected drift
- Identify root causes
- Recommend corrective actions
- Provide explainable reasoning
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from openai import OpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    OpenAI = None


@dataclass
class InvestigationContext:
    """Context for investigation"""
    model_id: str
    model_type: str
    trigger_event: str
    data_drift_report: Optional[Dict[str, Any]]
    concept_drift_report: Optional[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    recent_predictions: List[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]]
    historical_context: Optional[Dict[str, Any]]


@dataclass
class InvestigationResult:
    """Result of autonomous investigation"""
    model_id: str
    timestamp: str
    trigger_event: str
    investigation_summary: str
    root_cause_analysis: str
    recommended_actions: List[str]
    confidence_score: float
    llm_reasoning: str
    investigation_steps: List[str]
    severity_assessment: str


class InvestigationAgent:
    """Autonomous investigation agent powered by LLM"""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.3
    ):
        self.model_name = model_name
        self.temperature = temperature
        
        if openai_api_key:
            self.api_key = openai_api_key
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        if OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    def investigate(
        self,
        context: InvestigationContext
    ) -> InvestigationResult:
        """
        Conduct autonomous investigation of drift or performance issues
        
        Args:
            context: Investigation context with all relevant data
            
        Returns:
            InvestigationResult with analysis and recommendations
        """
        # Build investigation prompt
        prompt = self._build_investigation_prompt(context)
        
        # Get LLM analysis
        llm_response = self._query_llm(prompt)
        
        # Parse response
        result = self._parse_llm_response(context, llm_response)
        
        return result
    
    def _build_investigation_prompt(self, context: InvestigationContext) -> str:
        """Build comprehensive prompt for LLM investigation"""
        
        prompt = f"""You are an expert ML operations engineer investigating model performance issues. 

MODEL INFORMATION:
- Model ID: {context.model_id}
- Model Type: {context.model_type}
- Trigger Event: {context.trigger_event}

CURRENT PERFORMANCE:
{json.dumps(context.performance_metrics, indent=2)}

BASELINE PERFORMANCE:
{json.dumps(context.baseline_metrics, indent=2)}

"""
        
        if context.data_drift_report:
            prompt += f"""
DATA DRIFT ANALYSIS:
{json.dumps(context.data_drift_report, indent=2)}

"""
        
        if context.concept_drift_report:
            prompt += f"""
CONCEPT DRIFT ANALYSIS:
{json.dumps(context.concept_drift_report, indent=2)}

"""
        
        if context.feature_importance:
            prompt += f"""
FEATURE IMPORTANCE:
{json.dumps(context.feature_importance, indent=2)}

"""
        
        if context.historical_context:
            prompt += f"""
HISTORICAL CONTEXT:
{json.dumps(context.historical_context, indent=2)}

"""
        
        prompt += """
INVESTIGATION TASK:
Please analyze this situation and provide:

1. INVESTIGATION SUMMARY: A concise summary of what's happening
2. ROOT CAUSE ANALYSIS: Detailed analysis of why this drift is occurring. Consider:
   - Data quality issues (missing values, outliers, data corruption)
   - Distribution shifts in features
   - Changes in the underlying process being modeled
   - Seasonal patterns or external events
   - Data pipeline changes
   - Feature engineering issues
   - Model degradation over time

3. RECOMMENDED ACTIONS: Specific, actionable steps to address the issue. Prioritize by urgency. Include:
   - Immediate actions (if urgent)
   - Short-term fixes
   - Long-term solutions
   - Monitoring improvements

4. SEVERITY ASSESSMENT: Rate the severity (low/medium/high/critical) and explain why

5. CONFIDENCE SCORE: Your confidence in this analysis (0-1)

6. INVESTIGATION STEPS: List the steps you took to reach this conclusion

Format your response as JSON with these exact keys:
{
    "investigation_summary": "...",
    "root_cause_analysis": "...",
    "recommended_actions": ["action1", "action2", ...],
    "severity_assessment": "...",
    "confidence_score": 0.0-1.0,
    "investigation_steps": ["step1", "step2", ...]
}

Be specific and technical in your analysis. Use your expertise to identify subtle issues.
"""
        
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """Query LLM for investigation analysis"""
        if self.client is not None:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert ML operations engineer specializing in model monitoring, drift detection, and root cause analysis. Provide detailed, technical analysis with specific recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content
        else:
            # Fallback for when OpenAI is not available
            return self._generate_fallback_response()
    
    def _generate_fallback_response(self) -> str:
        """Generate a basic response when LLM is not available"""
        fallback = {
            "investigation_summary": "Drift detected but detailed analysis requires LLM integration.",
            "root_cause_analysis": "Unable to perform detailed root cause analysis without LLM. Manual review required.",
            "recommended_actions": [
                "Review data drift report manually",
                "Check data pipeline for changes",
                "Review recent model performance metrics",
                "Consider manual retraining if performance degraded significantly"
            ],
            "severity_assessment": "medium",
            "confidence_score": 0.3,
            "investigation_steps": [
                "Analyzed drift detection results",
                "Reviewed performance metrics",
                "Compared against baseline"
            ]
        }
        return json.dumps(fallback, indent=2)
    
    def _parse_llm_response(
        self,
        context: InvestigationContext,
        llm_response: str
    ) -> InvestigationResult:
        """Parse LLM response into structured result"""
        try:
            # Try to extract JSON from response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                parsed = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                parsed = {
                    "investigation_summary": llm_response[:200],
                    "root_cause_analysis": llm_response[200:800],
                    "recommended_actions": ["Manual review required"],
                    "severity_assessment": "medium",
                    "confidence_score": 0.5,
                    "investigation_steps": ["LLM response parsing failed"]
                }
        except json.JSONDecodeError:
            parsed = {
                "investigation_summary": "Failed to parse LLM response",
                "root_cause_analysis": llm_response,
                "recommended_actions": ["Manual investigation required"],
                "severity_assessment": "medium",
                "confidence_score": 0.3,
                "investigation_steps": ["Attempted to parse LLM response"]
            }
        
        return InvestigationResult(
            model_id=context.model_id,
            timestamp=datetime.utcnow().isoformat(),
            trigger_event=context.trigger_event,
            investigation_summary=parsed.get("investigation_summary", ""),
            root_cause_analysis=parsed.get("root_cause_analysis", ""),
            recommended_actions=parsed.get("recommended_actions", []),
            confidence_score=parsed.get("confidence_score", 0.5),
            llm_reasoning=llm_response,
            investigation_steps=parsed.get("investigation_steps", []),
            severity_assessment=parsed.get("severity_assessment", "medium")
        )
    
    def investigate_data_drift(
        self,
        model_id: str,
        drift_report: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> InvestigationResult:
        """Investigate data drift specifically"""
        prompt = f"""
You are investigating data drift for model {model_id}.

DRIFT REPORT:
{json.dumps(drift_report, indent=2)}

FEATURE IMPORTANCE:
{json.dumps(feature_importance, indent=2) if feature_importance else "Not available"}

Please analyze:
1. Which features are drifting most severely?
2. Are the drifting features important for model predictions?
3. What could cause these specific features to drift?
4. How should we prioritize fixing these issues?

Provide your analysis as JSON with keys:
- investigation_summary
- root_cause_analysis
- recommended_actions (prioritized list)
- severity_assessment
- confidence_score
- investigation_steps
"""
        
        llm_response = self._query_llm(prompt)
        context = InvestigationContext(
            model_id=model_id,
            model_type="unknown",
            trigger_event="data_drift_detected",
            data_drift_report=drift_report,
            concept_drift_report=None,
            performance_metrics={},
            baseline_metrics={},
            recent_predictions=[],
            feature_importance=feature_importance,
            historical_context=None
        )
        
        return self._parse_llm_response(context, llm_response)
    
    def investigate_concept_drift(
        self,
        model_id: str,
        drift_report: Dict[str, Any],
        performance_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> InvestigationResult:
        """Investigate concept drift specifically"""
        prompt = f"""
You are investigating concept drift for model {model_id}.

CONCEPT DRIFT REPORT:
{json.dumps(drift_report, indent=2)}

CURRENT PERFORMANCE:
{json.dumps(performance_metrics, indent=2)}

BASELINE PERFORMANCE:
{json.dumps(baseline_metrics, indent=2)}

Please analyze:
1. How much has performance degraded?
2. Is this gradual or sudden drift?
3. What type of concept drift is this (gradual, sudden, incremental, recurring)?
4. What could cause the relationship between features and target to change?
5. Should we retrain the model?

Provide your analysis as JSON with keys:
- investigation_summary
- root_cause_analysis
- recommended_actions (prioritized list)
- severity_assessment
- confidence_score
- investigation_steps
"""
        
        llm_response = self._query_llm(prompt)
        context = InvestigationContext(
            model_id=model_id,
            model_type="unknown",
            trigger_event="concept_drift_detected",
            data_drift_report=None,
            concept_drift_report=drift_report,
            performance_metrics=performance_metrics,
            baseline_metrics=baseline_metrics,
            recent_predictions=[],
            feature_importance=None,
            historical_context=None
        )
        
        return self._parse_llm_response(context, llm_response)
    
    def investigate_performance_degradation(
        self,
        model_id: str,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        recent_predictions: List[Dict[str, Any]]
    ) -> InvestigationResult:
        """Investigate general performance degradation"""
        prompt = f"""
You are investigating performance degradation for model {model_id}.

CURRENT METRICS:
{json.dumps(current_metrics, indent=2)}

BASELINE METRICS:
{json.dumps(baseline_metrics, indent=2)}

RECENT PREDICTIONS SAMPLE:
{json.dumps(recent_predictions[:10], indent=2)}

Please analyze:
1. Which metrics have degraded most?
2. Is this consistent with data drift or concept drift?
3. Are there patterns in the recent predictions that explain the degradation?
4. What immediate actions should be taken?

Provide your analysis as JSON with keys:
- investigation_summary
- root_cause_analysis
- recommended_actions (prioritized list)
- severity_assessment
- confidence_score
- investigation_steps
"""
        
        llm_response = self._query_llm(prompt)
        context = InvestigationContext(
            model_id=model_id,
            model_type="unknown",
            trigger_event="performance_degradation",
            data_drift_report=None,
            concept_drift_report=None,
            performance_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            recent_predictions=recent_predictions,
            feature_importance=None,
            historical_context=None
        )
        
        return self._parse_llm_response(context, llm_response)


class RuleBasedInvestigator:
    """Fallback rule-based investigator when LLM is not available"""
    
    def __init__(self):
        self.rules = self._load_rules()
    
    def _load_rules(self) -> List[Dict]:
        """Load investigation rules"""
        return [
            {
                "condition": "high_severity_data_drift",
                "actions": [
                    "Immediately investigate data pipeline",
                    "Check for data source changes",
                    "Review data quality metrics",
                    "Consider rolling back to previous model version"
                ],
                "severity": "high"
            },
            {
                "condition": "concept_drift_detected",
                "actions": [
                    "Analyze prediction distribution changes",
                    "Review model training data",
                    "Check for changes in target variable definition",
                    "Schedule model retraining"
                ],
                "severity": "medium"
            },
            {
                "condition": "performance_degradation",
                "actions": [
                    "Compare recent predictions with baseline",
                    "Check for prediction outliers",
                    "Review feature distributions",
                    "Monitor for continued degradation"
                ],
                "severity": "medium"
            },
            {
                "condition": "low_severity_drift",
                "actions": [
                    "Continue monitoring",
                    "Log drift for trend analysis",
                    "Review in next scheduled check"
                ],
                "severity": "low"
            }
        ]
    
    def investigate(
        self,
        context: InvestigationContext
    ) -> InvestigationResult:
        """Rule-based investigation"""
        applicable_rules = []
        
        # Check data drift severity
        if context.data_drift_report:
            severity = context.data_drift_report.get("overall_severity", "none")
            if severity in ["high", "critical"]:
                applicable_rules.append(self.rules[0])
            elif severity == "medium":
                applicable_rules.append(self.rules[1])
            elif severity == "low":
                applicable_rules.append(self.rules[3])
        
        # Check concept drift
        if context.concept_drift_report and context.concept_drift_report.get("drift_detected"):
            applicable_rules.append(self.rules[1])
        
        # Check performance degradation
        perf_diff = self._calculate_performance_difference(
            context.performance_metrics,
            context.baseline_metrics
        )
        if perf_diff > 0.1:
            applicable_rules.append(self.rules[2])
        
        # Combine recommendations
        all_actions = []
        max_severity = "low"
        
        for rule in applicable_rules:
            all_actions.extend(rule["actions"])
            if self._severity_rank(rule["severity"]) > self._severity_rank(max_severity):
                max_severity = rule["severity"]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_actions = []
        for action in all_actions:
            if action not in seen:
                seen.add(action)
                unique_actions.append(action)
        
        return InvestigationResult(
            model_id=context.model_id,
            timestamp=datetime.utcnow().isoformat(),
            trigger_event=context.trigger_event,
            investigation_summary=f"Rule-based investigation identified {len(applicable_rules)} applicable rules.",
            root_cause_analysis=self._generate_root_cause_analysis(context, applicable_rules),
            recommended_actions=unique_actions,
            confidence_score=0.6,
            llm_reasoning="Rule-based investigation (LLM not available)",
            investigation_steps=[f"Applied rule: {r['condition']}" for r in applicable_rules],
            severity_assessment=max_severity
        )
    
    def _calculate_performance_difference(
        self,
        current: Dict[str, float],
        baseline: Dict[str, float]
    ) -> float:
        """Calculate overall performance difference"""
        differences = []
        
        for key in baseline:
            if key in current:
                if key in ["accuracy", "r2", "f1"]:
                    # Higher is better
                    diff = baseline[key] - current[key]
                else:
                    # Lower is better (mse, mae, error_rate)
                    diff = current[key] - baseline[key]
                differences.append(abs(diff))
        
        return sum(differences) / len(differences) if differences else 0.0
    
    def _severity_rank(self, severity: str) -> int:
        """Get numeric rank for severity"""
        ranking = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return ranking.get(severity, 0)
    
    def _generate_root_cause_analysis(
        self,
        context: InvestigationContext,
        applicable_rules: List[Dict]
    ) -> str:
        """Generate root cause analysis based on rules"""
        analysis_parts = []
        
        if context.data_drift_report:
            analysis_parts.append(
                f"Data drift detected with severity {context.data_drift_report.get('overall_severity', 'unknown')}. "
                "This may indicate changes in input data distribution."
            )
        
        if context.concept_drift_report and context.concept_drift_report.get("drift_detected"):
            analysis_parts.append(
                f"Concept drift of type {context.concept_drift_report.get('drift_type', 'unknown')} detected. "
                "This indicates the relationship between features and target has changed."
            )
        
        if not analysis_parts:
            analysis_parts.append("No specific drift patterns identified. Continue monitoring.")
        
        return " ".join(analysis_parts)
