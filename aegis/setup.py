"""
Setup script for AEGIS
"""

from setuptools import setup, find_packages

setup(
    name="aegis-ml-reliability",
    version="1.0.0",
    description="Autonomous Model Reliability Engineer",
    author="AEGIS Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "evidently>=0.4.0",
        "river>=0.21.0",
        "nannyml>=0.13.0",
        "shap>=0.43.0",
        "mlflow>=2.10.0",
        "langgraph>=0.0.20",
        "google-generativeai>=0.3.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "streamlit>=1.31.0",
        "duckdb>=0.9.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "aegis-ui=ui.streamlit_app:main",
            "aegis-server=data_plane.serving:create_app",
        ],
    },
)
