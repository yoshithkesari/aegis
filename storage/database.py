"""
Storage Module for Monitoring Data

Handles persistence of:
- Model metadata
- Prediction logs
- Drift detection results
- Investigation reports
- Retraining history
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import os


@dataclass
class ModelMetadata:
    """Metadata for a monitored model"""
    model_id: str
    model_name: str
    model_type: str  # classification, regression, etc.
    model_path: str
    created_at: str
    last_updated: str
    is_active: bool
    baseline_metrics: Dict[str, float]
    feature_names: List[str]
    target_name: str


@dataclass
class PredictionLog:
    """Log of individual predictions"""
    id: Optional[int]
    model_id: str
    timestamp: str
    features: Dict[str, Any]
    prediction: Any
    actual_value: Optional[Any]
    prediction_probability: Optional[float]
    metadata: Optional[Dict[str, Any]]


@dataclass
class DriftDetectionResult:
    """Result of drift detection"""
    id: Optional[int]
    model_id: str
    timestamp: str
    drift_type: str  # data_drift, concept_drift
    drift_detected: bool
    severity: str
    metrics: Dict[str, Any]
    summary: str
    recommendations: List[str]


@dataclass
class InvestigationReport:
    """Report from autonomous investigation"""
    id: Optional[int]
    model_id: str
    timestamp: str
    trigger_event: str
    investigation_summary: str
    root_cause_analysis: str
    recommended_actions: List[str]
    confidence_score: float
    llm_reasoning: str


@dataclass
class RetrainingRecord:
    """Record of retraining operations"""
    id: Optional[int]
    model_id: str
    timestamp: str
    trigger_reason: str
    old_version: str
    new_version: str
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    status: str  # started, completed, failed
    duration_seconds: Optional[float]
    data_used: Dict[str, Any]


class MonitoringDatabase:
    """Database for storing monitoring data"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self._initialize_database()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _initialize_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    baseline_metrics TEXT NOT NULL,
                    feature_names TEXT NOT NULL,
                    target_name TEXT NOT NULL
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    features TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    actual_value TEXT,
                    prediction_probability REAL,
                    metadata TEXT,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            """)
            
            # Drift detection results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    drift_type TEXT NOT NULL,
                    drift_detected BOOLEAN NOT NULL,
                    severity TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            """)
            
            # Investigation reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS investigations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    trigger_event TEXT NOT NULL,
                    investigation_summary TEXT NOT NULL,
                    root_cause_analysis TEXT NOT NULL,
                    recommended_actions TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    llm_reasoning TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            """)
            
            # Retraining records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retraining_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    old_version TEXT NOT NULL,
                    new_version TEXT NOT NULL,
                    training_metrics TEXT NOT NULL,
                    validation_metrics TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_seconds REAL,
                    data_used TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_model_id 
                ON predictions(model_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
                ON predictions(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_drift_results_model_id 
                ON drift_results(model_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_drift_results_timestamp 
                ON drift_results(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_investigations_model_id 
                ON investigations(model_id)
            """)
    
    # Model operations
    def register_model(self, metadata: ModelMetadata) -> bool:
        """Register a new model for monitoring"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO models 
                    (model_id, model_name, model_type, model_path, created_at, 
                     last_updated, is_active, baseline_metrics, feature_names, target_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.model_id,
                    metadata.model_name,
                    metadata.model_type,
                    metadata.model_path,
                    metadata.created_at,
                    metadata.last_updated,
                    metadata.is_active,
                    json.dumps(metadata.baseline_metrics),
                    json.dumps(metadata.feature_names),
                    metadata.target_name
                ))
                return True
            except sqlite3.IntegrityError:
                return False
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return ModelMetadata(
                model_id=row['model_id'],
                model_name=row['model_name'],
                model_type=row['model_type'],
                model_path=row['model_path'],
                created_at=row['created_at'],
                last_updated=row['last_updated'],
                is_active=bool(row['is_active']),
                baseline_metrics=json.loads(row['baseline_metrics']),
                feature_names=json.loads(row['feature_names']),
                target_name=row['target_name']
            )
    
    def list_models(self, active_only: bool = True) -> List[ModelMetadata]:
        """List all models"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM models"
            if active_only:
                query += " WHERE is_active = 1"
            cursor.execute(query)
            
            models = []
            for row in cursor.fetchall():
                models.append(ModelMetadata(
                    model_id=row['model_id'],
                    model_name=row['model_name'],
                    model_type=row['model_type'],
                    model_path=row['model_path'],
                    created_at=row['created_at'],
                    last_updated=row['last_updated'],
                    is_active=bool(row['is_active']),
                    baseline_metrics=json.loads(row['baseline_metrics']),
                    feature_names=json.loads(row['feature_names']),
                    target_name=row['target_name']
                ))
            return models
    
    def update_model(self, model_id: str, **kwargs) -> bool:
        """Update model metadata"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            valid_fields = ['model_name', 'model_type', 'model_path', 'last_updated', 
                          'is_active', 'baseline_metrics', 'feature_names', 'target_name']
            
            updates = []
            values = []
            
            for field, value in kwargs.items():
                if field in valid_fields:
                    if field in ['baseline_metrics', 'feature_names']:
                        value = json.dumps(value)
                    updates.append(f"{field} = ?")
                    values.append(value)
            
            if not updates:
                return False
            
            values.append(model_id)
            query = f"UPDATE models SET {', '.join(updates)} WHERE model_id = ?"
            
            cursor.execute(query, values)
            return cursor.rowcount > 0
    
    # Prediction operations
    def log_prediction(self, log: PredictionLog) -> int:
        """Log a prediction"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions 
                (model_id, timestamp, features, prediction, actual_value, 
                 prediction_probability, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                log.model_id,
                log.timestamp,
                json.dumps(log.features),
                json.dumps(log.prediction),
                json.dumps(log.actual_value) if log.actual_value else None,
                log.prediction_probability,
                json.dumps(log.metadata) if log.metadata else None
            ))
            return cursor.lastrowid
    
    def get_predictions(
        self,
        model_id: str,
        limit: int = 1000,
        offset: int = 0,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[PredictionLog]:
        """Get predictions for a model"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM predictions WHERE model_id = ?"
            params = [model_id]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            
            predictions = []
            for row in cursor.fetchall():
                predictions.append(PredictionLog(
                    id=row['id'],
                    model_id=row['model_id'],
                    timestamp=row['timestamp'],
                    features=json.loads(row['features']),
                    prediction=json.loads(row['prediction']),
                    actual_value=json.loads(row['actual_value']) if row['actual_value'] else None,
                    prediction_probability=row['prediction_probability'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else None
                ))
            return predictions
    
    def get_predictions_for_analysis(
        self,
        model_id: str,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """Get predictions formatted for drift analysis"""
        predictions = self.get_predictions(model_id, limit=sample_size)
        
        features_list = []
        predictions_list = []
        actuals_list = []
        
        for pred in predictions:
            features_list.append(pred.features)
            predictions_list.append(pred.prediction)
            if pred.actual_value is not None:
                actuals_list.append(pred.actual_value)
        
        return {
            'features': features_list,
            'predictions': predictions_list,
            'actuals': actuals_list
        }
    
    # Drift detection operations
    def save_drift_result(self, result: DriftDetectionResult) -> int:
        """Save drift detection result"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO drift_results 
                (model_id, timestamp, drift_type, drift_detected, severity, 
                 metrics, summary, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.model_id,
                result.timestamp,
                result.drift_type,
                result.drift_detected,
                result.severity,
                json.dumps(result.metrics),
                result.summary,
                json.dumps(result.recommendations)
            ))
            return cursor.lastrowid
    
    def get_drift_results(
        self,
        model_id: str,
        limit: int = 100,
        drift_type: Optional[str] = None
    ) -> List[DriftDetectionResult]:
        """Get drift detection results"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM drift_results WHERE model_id = ?"
            params = [model_id]
            
            if drift_type:
                query += " AND drift_type = ?"
                params.append(drift_type)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append(DriftDetectionResult(
                    id=row['id'],
                    model_id=row['model_id'],
                    timestamp=row['timestamp'],
                    drift_type=row['drift_type'],
                    drift_detected=bool(row['drift_detected']),
                    severity=row['severity'],
                    metrics=json.loads(row['metrics']),
                    summary=row['summary'],
                    recommendations=json.loads(row['recommendations'])
                ))
            return results
    
    # Investigation operations
    def save_investigation(self, report: InvestigationReport) -> int:
        """Save investigation report"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO investigations 
                (model_id, timestamp, trigger_event, investigation_summary, 
                 root_cause_analysis, recommended_actions, confidence_score, llm_reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.model_id,
                report.timestamp,
                report.trigger_event,
                report.investigation_summary,
                report.root_cause_analysis,
                json.dumps(report.recommended_actions),
                report.confidence_score,
                report.llm_reasoning
            ))
            return cursor.lastrowid
    
    def get_investigations(self, model_id: str, limit: int = 50) -> List[InvestigationReport]:
        """Get investigation reports"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM investigations 
                WHERE model_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (model_id, limit))
            
            reports = []
            for row in cursor.fetchall():
                reports.append(InvestigationReport(
                    id=row['id'],
                    model_id=row['model_id'],
                    timestamp=row['timestamp'],
                    trigger_event=row['trigger_event'],
                    investigation_summary=row['investigation_summary'],
                    root_cause_analysis=row['root_cause_analysis'],
                    recommended_actions=json.loads(row['recommended_actions']),
                    confidence_score=row['confidence_score'],
                    llm_reasoning=row['llm_reasoning']
                ))
            return reports
    
    # Retraining operations
    def save_retraining_record(self, record: RetrainingRecord) -> int:
        """Save retraining record"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO retraining_records 
                (model_id, timestamp, trigger_reason, old_version, new_version, 
                 training_metrics, validation_metrics, status, duration_seconds, data_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.model_id,
                record.timestamp,
                record.trigger_reason,
                record.old_version,
                record.new_version,
                json.dumps(record.training_metrics),
                json.dumps(record.validation_metrics),
                record.status,
                record.duration_seconds,
                json.dumps(record.data_used)
            ))
            return cursor.lastrowid
    
    def get_retraining_history(self, model_id: str, limit: int = 50) -> List[RetrainingRecord]:
        """Get retraining history"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM retraining_records 
                WHERE model_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (model_id, limit))
            
            records = []
            for row in cursor.fetchall():
                records.append(RetrainingRecord(
                    id=row['id'],
                    model_id=row['model_id'],
                    timestamp=row['timestamp'],
                    trigger_reason=row['trigger_reason'],
                    old_version=row['old_version'],
                    new_version=row['new_version'],
                    training_metrics=json.loads(row['training_metrics']),
                    validation_metrics=json.loads(row['validation_metrics']),
                    status=row['status'],
                    duration_seconds=row['duration_seconds'],
                    data_used=json.loads(row['data_used'])
                ))
            return records
    
    # Utility operations
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to manage database size"""
        cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete old predictions (keep drift results and investigations)
            cursor.execute(
                "DELETE FROM predictions WHERE timestamp < ?",
                (cutoff_date,)
            )
            
            deleted_predictions = cursor.rowcount
            
            return {
                'deleted_predictions': deleted_predictions,
                'cutoff_date': cutoff_date
            }
    
    def get_model_statistics(self, model_id: str) -> Dict[str, Any]:
        """Get statistics for a model"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute(
                "SELECT COUNT(*) FROM predictions WHERE model_id = ?",
                (model_id,)
            )
            total_predictions = cursor.fetchone()[0]
            
            # Drift detections
            cursor.execute(
                "SELECT COUNT(*) FROM drift_results WHERE model_id = ? AND drift_detected = 1",
                (model_id,)
            )
            drift_detections = cursor.fetchone()[0]
            
            # Investigations
            cursor.execute(
                "SELECT COUNT(*) FROM investigations WHERE model_id = ?",
                (model_id,)
            )
            investigations = cursor.fetchone()[0]
            
            # Retrainings
            cursor.execute(
                "SELECT COUNT(*) FROM retraining_records WHERE model_id = ?",
                (model_id,)
            )
            retrainings = cursor.fetchone()[0]
            
            return {
                'total_predictions': total_predictions,
                'drift_detections': drift_detections,
                'investigations': investigations,
                'retrainings': retrainings
            }
