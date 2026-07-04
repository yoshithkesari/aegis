"""
Configuration Management Module

Handles:
- Loading configuration from environment variables and config files
- Managing settings for all components
- Validation of configuration values
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from dotenv import load_dotenv


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "sqlite:///monitoring.db"
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: Optional[str] = None
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 60


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    interval_seconds: int = 300
    drift_threshold: float = 0.05
    performance_threshold: float = 0.1
    confidence_threshold: float = 0.95
    window_size: int = 100
    min_samples: int = 50


@dataclass
class RetrainingConfig:
    """Retraining configuration"""
    data_path: str = "data/"
    model_save_path: str = "models/"
    max_iterations: int = 5
    early_stop_patience: int = 3
    validation_split: float = 0.2
    min_improvement: float = 0.01
    max_training_time: int = 3600


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    sample_size: int = 1000
    timeout_seconds: int = 300
    performance_threshold: float = 0.05
    consistency_threshold: float = 0.95
    latency_threshold_ms: float = 100.0
    error_rate_threshold: float = 0.01


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    log_file: str = "logs/monitoring.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size_mb: int = 100
    backup_count: int = 5


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class Config:
    """Main configuration class"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        load_dotenv()
        
        return cls(
            database=DatabaseConfig(
                url=os.getenv("DATABASE_URL", "sqlite:///monitoring.db"),
                pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
                max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
                echo=os.getenv("DB_ECHO", "false").lower() == "true"
            ),
            openai=OpenAIConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
                timeout=int(os.getenv("OPENAI_TIMEOUT", "60"))
            ),
            monitoring=MonitoringConfig(
                interval_seconds=int(os.getenv("MONITORING_INTERVAL", "300")),
                drift_threshold=float(os.getenv("DRIFT_THRESHOLD", "0.05")),
                performance_threshold=float(os.getenv("PERFORMANCE_THRESHOLD", "0.1")),
                confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.95")),
                window_size=int(os.getenv("WINDOW_SIZE", "100")),
                min_samples=int(os.getenv("MIN_SAMPLES", "50"))
            ),
            retraining=RetrainingConfig(
                data_path=os.getenv("RETRAINING_DATA_PATH", "data/"),
                model_save_path=os.getenv("RETRAINING_MODEL_SAVE_PATH", "models/"),
                max_iterations=int(os.getenv("RETRAINING_MAX_ITERATIONS", "5")),
                early_stop_patience=int(os.getenv("RETRAINING_EARLY_STOP_PATIENCE", "3")),
                validation_split=float(os.getenv("RETRAINING_VALIDATION_SPLIT", "0.2")),
                min_improvement=float(os.getenv("RETRAINING_MIN_IMPROVEMENT", "0.01")),
                max_training_time=int(os.getenv("RETRAINING_MAX_TRAINING_TIME", "3600"))
            ),
            deployment=DeploymentConfig(
                sample_size=int(os.getenv("DEPLOYMENT_SAMPLE_SIZE", "1000")),
                timeout_seconds=int(os.getenv("DEPLOYMENT_TIMEOUT", "300")),
                performance_threshold=float(os.getenv("DEPLOYMENT_PERFORMANCE_THRESHOLD", "0.05")),
                consistency_threshold=float(os.getenv("DEPLOYMENT_CONSISTENCY_THRESHOLD", "0.95")),
                latency_threshold_ms=float(os.getenv("DEPLOYMENT_LATENCY_THRESHOLD_MS", "100.0")),
                error_rate_threshold=float(os.getenv("DEPLOYMENT_ERROR_RATE_THRESHOLD", "0.01"))
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                log_file=os.getenv("LOG_FILE", "logs/monitoring.log"),
                log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                max_file_size_mb=int(os.getenv("LOG_MAX_FILE_SIZE_MB", "100")),
                backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
            ),
            api=APIConfig(
                host=os.getenv("API_HOST", "0.0.0.0"),
                port=int(os.getenv("API_PORT", "8000")),
                workers=int(os.getenv("API_WORKERS", "1")),
                reload=os.getenv("API_RELOAD", "false").lower() == "true",
                cors_origins=os.getenv("API_CORS_ORIGINS", "*").split(",")
            )
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            database=DatabaseConfig(**config_dict.get('database', {})),
            openai=OpenAIConfig(**config_dict.get('openai', {})),
            monitoring=MonitoringConfig(**config_dict.get('monitoring', {})),
            retraining=RetrainingConfig(**config_dict.get('retraining', {})),
            deployment=DeploymentConfig(**config_dict.get('deployment', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            api=APIConfig(**config_dict.get('api', {}))
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'database': self.database.__dict__,
            'openai': self.openai.__dict__,
            'monitoring': self.monitoring.__dict__,
            'retraining': self.retraining.__dict__,
            'deployment': self.deployment.__dict__,
            'logging': self.logging.__dict__,
            'api': self.api.__dict__
        }
        
        # Remove None values
        config_dict = {k: {kk: vv for kk, vv in v.items() if vv is not None} 
                      for k, v in config_dict.items()}
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration values"""
        errors = []
        warnings = []
        
        # Validate OpenAI config
        if not self.openai.api_key:
            errors.append("OpenAI API key is required for LLM-powered investigation")
        
        # Validate monitoring config
        if self.monitoring.drift_threshold < 0 or self.monitoring.drift_threshold > 1:
            errors.append("Drift threshold must be between 0 and 1")
        
        if self.monitoring.performance_threshold < 0:
            errors.append("Performance threshold must be positive")
        
        # Validate retraining config
        if not os.path.exists(self.retraining.data_path):
            warnings.append(f"Retraining data path {self.retraining.data_path} does not exist")
        
        if not os.path.exists(self.retraining.model_save_path):
            warnings.append(f"Model save path {self.retraining.model_save_path} does not exist")
        
        # Validate logging config
        log_dir = os.path.dirname(self.logging.log_file)
        if log_dir and not os.path.exists(log_dir):
            warnings.append(f"Log directory {log_dir} does not exist")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


def get_config() -> Config:
    """Get configuration (loads from environment by default)"""
    config_file = os.getenv("CONFIG_FILE")
    
    if config_file and os.path.exists(config_file):
        return Config.from_yaml(config_file)
    else:
        return Config.from_env()


def setup_logging(config: LoggingConfig):
    """Setup logging based on configuration"""
    import logging
    from logging.handlers import RotatingFileHandler
    
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(config.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.log_format,
        handlers=[
            RotatingFileHandler(
                config.log_file,
                maxBytes=config.max_file_size_mb * 1024 * 1024,
                backupCount=config.backup_count
            ),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
