"""
Agentic Model Monitoring & Drift Detection

An autonomous AI operations engineer that continuously monitors deployed machine learning models,
detects data drift and concept drift, identifies root causes, validates model health,
recommends corrective actions, automatically retrains models when necessary,
and verifies deployment success.
"""

__version__ = "1.0.0"

from core import ModelMonitor, MonitoringOrchestrator, MonitoringConfig
from storage import MonitoringDatabase
from drift_detection import DataDriftDetector, ConceptDriftDetector
from agent import InvestigationAgent
from retraining import RetrainingPipeline
from deployment import DeploymentVerifier

__all__ = [
    'ModelMonitor',
    'MonitoringOrchestrator',
    'MonitoringConfig',
    'MonitoringDatabase',
    'DataDriftDetector',
    'ConceptDriftDetector',
    'InvestigationAgent',
    'RetrainingPipeline',
    'DeploymentVerifier'
]
