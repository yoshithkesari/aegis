# AEGIS - Autonomous Model Reliability

An on-call SRE for your production ML models. AEGIS treats model degradation like an incident — it investigates the root cause, remediates under guardrails, validates without waiting for labels, redeploys, and writes the postmortem.

## The Problem

Models rot in production — and the tooling stops at the alarm. As the real world shifts (new fraud patterns, changing behavior, upstream pipeline changes), a model's inputs drift (data/covariate drift) or the input→output relationship changes (concept drift). The model keeps returning confident predictions while quietly getting worse, and organizations often notice only after the loss is booked.

## The Solution

AEGIS is an autonomous ML-reliability engineer. It detects, opens an incident, investigates the root cause, decides a remediation under explicit risk guardrails, retrains, validates without waiting for labels, redeploys via canary, verifies, and writes the postmortem — escalating to a human only when risk is high.

### Four Differentiators

1. **Incidents & Postmortems, Not Alerts** — Every judge has lived a PagerDuty incident. AEGIS opens an incident, fixes it, and files a human-readable postmortem — instantly legible, instantly differentiated.

2. **Risk-Gated Autonomy** — A decision matrix auto-fixes the safe cases and escalates the dangerous ones with a full diagnosis. Autonomy you can defend, not "it ships models on its own."

3. **Label-Free Validation** — Estimates a retrain's live performance without ground truth (NannyML CBPE), so the deploy gate works even when labels are weeks away.

4. **Provable Root Cause** — A drift-injection testbed with known causes lets you score diagnosis accuracy objectively — the credibility multiplier most teams skip.

## Architecture

Two loops — and only one of them is agentic.

### Outer Loop: Deterministic Controller
- **Role**: Drives the incident state machine; executes every write action
- **LLM**: None. The retrain trigger and deploy gate are numeric thresholds
- **Why**: Must be predictable, idempotent, auditable — it's the safety-critical path

### Inner Loop: Agentic Investigation
- **Role**: Root-cause investigation only. Fires when an incident opens
- **Loop**: hypothesize → call a diagnostic tool → observe → refine → conclude
- **Access**: LLM is read-only — it can investigate, it cannot deploy

**The LLM recommends. The controller acts.** That read/write split is a security control: even a prompt-injected or hallucinating model physically cannot deploy anything.

## Tech Stack (All Free)

| Layer | Choice | Why | Cost |
|-------|--------|-----|------|
| Monitored model | XGBoost / scikit-learn | Trivial on purpose — it's the patient, not the product | $0 |
| Data / feature drift | Evidently | PSI, KS, Wasserstein, JS-divergence, reports | $0 |
| Streaming / concept drift | River | ADWIN, DDM, Page-Hinkley — real online detectors | $0 |
| Label-free validation | NannyML | CBPE / DLE — the hero; validates without labels | $0 |
| Drift attribution | SHAP | Per-feature contribution for root cause | $0 |
| Registry / champion-challenger | MLflow | Versioning, stage transitions, rollback | $0 |
| Orchestration | LangGraph | Models both the state machine and the agent sub-loop | $0* |
| Reasoning + postmortem | Gemini Flash (free tier) | Small, fast, hosted — zero local weights | Free tier |
| Serving + routing | FastAPI | Prediction endpoint + shadow/canary | $0 |
| Store | SQLite + DuckDB | Zero-ops incident + metrics store | $0 |
| Dashboard / UI | Streamlit | Incident feed + live scoreboard, fast to build | $0 |

* LangGraph is free — just don't enable LangSmith tracing (paid + sends traces off-machine).

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd aegis

# Install (editable, from pyproject.toml). Extras: [hosted] Gemini, [dev] tests
pip install -e ".[hosted,dev]"

# Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Generate training data
python -c "from aegis.eval.scenarios import ScenarioRunner; ScenarioRunner().run_all_scenarios()"
```

## Running the Demo

### Option 1: Streamlit UI (Recommended)

```bash
streamlit run src/aegis/ui/streamlit_app.py
```

This launches the professional white-background UI with the two-panel narrative:
- **Left Panel**: Problem (incumbent detection-only behavior)
- **Right Panel**: Solution (AEGIS autonomous remediation)

Click "Run Demo" to see the three-act demonstration:
1. **Act 1**: Inject drift → model degrades → incumbent stops at detection
2. **Act 2**: AEGIS runs the loop → investigate → diagnose → retrain → validate → canary → promote
3. **Act 3**: Scoreboard shows measured results

### Option 2: API Server

```bash
python -m aegis.data_plane.serving
```

This starts the FastAPI model server with champion/challenger routing.

### Option 3: Eval Harness

```bash
python -c "
from aegis.eval.scenarios import ScenarioRunner
from aegis.eval.scoreboard import Scoreboard, EvalHarness
from aegis.control_plane.detectors import DetectorSuite
from aegis.agent.investigator import Investigator

# Run scenarios
runner = ScenarioRunner()
scenarios = runner.run_all_scenarios()

# Evaluate
scoreboard = Scoreboard()
harness = EvalHarness(scoreboard)
detector = DetectorSuite()
investigator = Investigator()

results = harness.run_all_scenarios(scenarios, detector, investigator, None)
scoreboard.print_scoreboard()
"
```

## Free Deployment

### Streamlit Community Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and connect your GitHub repo
4. Set the main file path to `src/aegis/ui/streamlit_app.py`
5. In Secrets, add `GEMINI_API_KEY` (get it from [aistudio.google.com](https://aistudio.google.com/app/apikey))
6. Deploy

**Cost**: $0 (free tier)
**RAM**: Up to available on free tier
**Model weights**: 0 (Gemini runs on Google's servers)

### Hugging Face Spaces

1. Push your code to GitHub
2. Create a new Space on Hugging Face
3. Choose "Streamlit" as the SDK
4. Connect your GitHub repo
5. Add `GEMINI_API_KEY` in Repository Secrets
6. Deploy

**Cost**: $0 (free tier)
**RAM**: 16 GB on free tier

## Security & Privacy

### Guaranteed No Charges
- Attach no credit card anywhere
- Run locally; avoid cloud free-tiers that bill for forgotten resources
- Free-tier LLM keys rate-limit (429), never bill, if no billing is linked
- Bounded loops: max tool-calls/incident, max retrains/day, kill-switch

### No Data Leakage
- Local LLM = no egress — features and data never leave the machine
- Hosted fallback: send aggregated stats only, never raw rows or PII
- Bind MLflow / Streamlit / FastAPI to 127.0.0.1, not 0.0.0.0
- Demo uses synthetic data — no real PII exists to leak

### Deployment Security
- **Read/write separation** — The LLM can't deploy, only investigate
- **Data-poisoning defense** — The risk-gate blocks retrains that would regress
- **Canary + auto-rollback** — Blast radius is bounded
- **Full audit trail** — Every controller action is logged

### Adversarial Drift Defense
An attacker could deliberately inject drift to trick the agent into auto-retraining on poisoned data. Defenses:
- Label-free estimate blocks regressive retrains
- Anomaly checks guard the retrain window
- Retrain-frequency cap + cooldown
- Previous champion always kept for instant rollback
- Suspicious patterns escalate to human instead of auto-fixing

## Project Structure

```
aegis/
├── pyproject.toml             # Single source of deps + entrypoints
├── README.md  .env.example  LICENSE
├── src/aegis/
│   ├── config.py              # One settings module (env / .env)
│   ├── data_plane/            # The "patient": replay, drift injection, serving
│   │   ├── replayer.py  drift_injector.py  serving.py
│   ├── control_plane/         # Deterministic, write-authority
│   │   ├── controller.py      # Incident state machine
│   │   ├── detectors.py       # Evidently + River
│   │   ├── risk_gate.py       # Decision matrix
│   │   └── remediation.py     # Retrain / canary / rollback
│   ├── orchestration/         # LangGraph state machine (+ plain-Python fallback)
│   │   └── graph.py
│   ├── agent/                 # Inner loop: read-only investigation
│   │   ├── investigator.py  tools.py
│   ├── reasoning/             # Pluggable LLM: Gemini → Ollama → heuristic
│   │   ├── base.py  gemini.py  local.py
│   ├── validation/            # Label-free gate + invariance suite
│   │   ├── label_free.py  invariance_suite.py
│   ├── stores/                # Durable: registry (MLflow) · incidents (SQLite) · metrics (DuckDB)
│   │   ├── registry.py  incidents.py  metrics.py
│   ├── eval/                  # Injected drift + scoreboard
│   │   ├── scenarios.py  scoreboard.py
│   └── ui/streamlit_app.py    # Two-panel narrative
├── tests/                     # risk_gate · controller · read/write split
├── data/  models/  artifacts/ # Data, model artifacts, precomputed outputs
```

## Running the Tests

```bash
pip install -e ".[dev]"
pytest            # 15 tests: risk-gate logic, state-machine loop, read/write split
```

## License

MIT

## Acknowledgments

Built for the MLOps Autonomous Reliability Hackathon.

The architecture is inspired by production ML reliability best practices and the need for autonomous remediation that is both safe and defensible.
