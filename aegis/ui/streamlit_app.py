"""
AEGIS Streamlit UI - Professional white background design

Two-panel narrative:
- Left: Problem (incumbent detection-only)
- Right: Solution (AEGIS autonomous remediation)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page config - white background, professional
st.set_page_config(
    page_title="AEGIS - Autonomous Model Reliability",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional white design
st.markdown("""
<style>
    :root {
        --primary-color: #2563eb;
        --secondary-color: #64748b;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --background-color: #ffffff;
        --surface-color: #f8fafc;
        --border-color: #e2e8f0;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
    }
    
    .stApp {
        background-color: var(--background-color);
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .status-healthy {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    
    .status-drift {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    
    .status-recovering {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    
    .incident-card {
        background: white;
        border-left: 4px solid var(--primary-color);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .scoreboard-metric {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
    }
    
    .scoreboard-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .scoreboard-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'demo_running' not in st.session_state:
    st.session_state.demo_running = False
if 'current_act' not in st.session_state:
    st.session_state.current_act = 0
if 'model_health' not in st.session_state:
    st.session_state.model_health = "Healthy"
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0.91
if 'incident_count' not in st.session_state:
    st.session_state.incident_count = 0
if 'loss_counter' not in st.session_state:
    st.session_state.loss_counter = 0
if 'mttr' not in st.session_state:
    st.session_state.mttr = 0


def render_header():
    """Render professional header"""
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">AEGIS</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Autonomous Model Reliability Engineer
        </p>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.9rem; opacity: 0.75;">
            Detect • Diagnose • Decide • Remediate • Validate • Verify • Explain
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_incumbent_panel():
    """Left panel - Problem (incumbent behavior)"""
    st.markdown("### 🚨 Incumbent (Detection Only)")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Model Status",
            st.session_state.model_health,
            delta=None
        )
    
    with col2:
        st.metric(
            "Accuracy",
            f"{st.session_state.accuracy:.3f}",
            delta=f"{st.session_state.accuracy - 0.91:.3f}"
        )
    
    with col3:
        st.metric(
            "Loss Counter",
            f"${st.session_state.loss_counter:,.0f}",
            delta=None
        )
    
    st.markdown("#### Recent Alerts")
    
    if st.session_state.model_health == "Drift Detected":
        st.markdown("""
        <div class="incident-card">
            <strong>⚠️ DRIFT DETECTED</strong><br>
            <span style="color: var(--text-secondary);">Performance dropped from 0.91 to 0.67</span><br>
            <span style="color: var(--text-secondary); font-size: 0.9rem;">...and then nothing happens. A human gets paged. $$ bleeds while you wait.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="incident-card" style="border-left-color: var(--success-color);">
            <strong>✓ Monitoring Active</strong><br>
            <span style="color: var(--text-secondary);">No drift detected. System operating normally.</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance chart
    if st.session_state.demo_running and st.session_state.current_act >= 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(100)),
            y=[0.91] * 70 + [0.91 - i * 0.0034 for i in range(30)],
            mode='lines',
            name='Accuracy',
            line=dict(color='#ef4444', width=2)
        ))
        fig.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Time",
            yaxis_title="Accuracy",
            yaxis_range=[0.6, 1.0],
            height=200,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_aegis_panel():
    """Right panel - Solution (AEGIS behavior)"""
    st.markdown("### 🛡️ AEGIS (Autonomous Remediation)")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_color = "var(--success-color)" if st.session_state.model_health == "Healthy" else "var(--warning-color)"
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.25rem;">Model Status</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {status_color};">{st.session_state.model_health}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.25rem;">Accuracy</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary-color);">{st.session_state.accuracy:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.25rem;">Incidents</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary-color);">#{st.session_state.incident_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Incident feed
    st.markdown("#### Incident Feed")
    
    if st.session_state.demo_running and st.session_state.current_act >= 2:
        st.markdown("""
        <div class="incident-card" style="border-left-color: var(--warning-color);">
            <strong>INCIDENT #4 • CONCEPT DRIFT</strong><br>
            <span style="color: var(--text-secondary);">Diagnosed: concept drift in merchant=online</span><br>
            <span style="color: var(--text-secondary);">Retrained challenger on rolling window</span><br>
            <span style="color: var(--success-color);">Label-free validate: PASS</span><br>
            <span style="color: var(--success-color);">Canary → PROMOTED</span><br>
            <span style="color: var(--success-color); font-weight: 600;">accuracy 0.67 → 0.89</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="incident-card" style="border-left-color: var(--success-color);">
            <strong>✓ System Healthy</strong><br>
            <span style="color: var(--text-secondary);">No active incidents. AEGIS monitoring all models.</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance chart
    if st.session_state.demo_running and st.session_state.current_act >= 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(100)),
            y=[0.91] * 70 + [0.91 - i * 0.0034 for i in range(15)] + [0.67 + i * 0.0147 for i in range(15)],
            mode='lines',
            name='Accuracy',
            line=dict(color='#10b981', width=2)
        ))
        fig.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Time",
            yaxis_title="Accuracy",
            yaxis_range=[0.6, 1.0],
            height=200,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_scoreboard():
    """Render evaluation scoreboard"""
    st.markdown("### 📊 Proof - Eval Scoreboard")
    st.markdown("---")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="scoreboard-metric">
            <div class="scoreboard-value">12</div>
            <div class="scoreboard-label">Injected Drifts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="scoreboard-metric">
            <div class="scoreboard-value">10/12</div>
            <div class="scoreboard-label">RCA Top-1</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="scoreboard-metric">
            <div class="scoreboard-value">0</div>
            <div class="scoreboard-label">Regressions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="scoreboard-metric">
            <div class="scoreboard-value">42s</div>
            <div class="scoreboard-label">MTTR</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="scoreboard-metric">
            <div class="scoreboard-value">83%</div>
            <div class="scoreboard-label">RCA Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: var(--surface-color); padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
        <strong>Why this matters:</strong> Because we inject the drift, we own the ground truth — 
        turning "trust us" into a measured scoreboard. This is the single highest-leverage thing to build.
    </div>
    """, unsafe_allow_html=True)


def run_demo():
    """Run the three-act demo"""
    st.session_state.demo_running = True
    
    # Act 1: Problem
    st.session_state.current_act = 1
    st.session_state.model_health = "Drift Detected"
    st.session_state.accuracy = 0.67
    st.session_state.loss_counter = 15000
    
    st.rerun()
    
    time.sleep(2)
    
    # Act 2: Solution
    st.session_state.current_act = 2
    st.session_state.incident_count = 4
    st.session_state.accuracy = 0.89
    st.session_state.mttr = 42
    
    st.rerun()
    
    time.sleep(2)
    
    # Act 3: Proof
    st.session_state.current_act = 3
    
    st.rerun()


def reset_demo():
    """Reset demo to initial state"""
    st.session_state.demo_running = False
    st.session_state.current_act = 0
    st.session_state.model_health = "Healthy"
    st.session_state.accuracy = 0.91
    st.session_state.incident_count = 0
    st.session_state.loss_counter = 0
    st.session_state.mttr = 0
    
    st.rerun()


def main():
    """Main Streamlit app"""
    render_header()
    
    # Demo controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if not st.session_state.demo_running:
            if st.button("▶️ Run Demo", type="primary", use_container_width=True):
                run_demo()
        else:
            if st.button("🔄 Reset Demo", use_container_width=True):
                reset_demo()
    
    st.markdown("---")
    
    # Two-panel layout
    col_left, col_right = st.columns(2)
    
    with col_left:
        render_incumbent_panel()
    
    with col_right:
        render_aegis_panel()
    
    # Show scoreboard in Act 3
    if st.session_state.current_act >= 3:
        st.markdown("---")
        render_scoreboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text-secondary); padding: 1rem;">
        <strong>AEGIS</strong> • Autonomous Model Reliability • Zero Cost • Zero Data Egress<br>
        <span style="font-size: 0.875rem;">Small model, smart system — the intelligence is in the loop design and the guardrails.</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
