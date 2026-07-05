# AEGIS — Autonomous Model Reliability Engineer

> An on-call SRE for your production ML models. AEGIS treats model degradation
> like an incident — it detects, investigates the root cause, remediates under
> guardrails, **validates without waiting for labels**, redeploys via canary,
> verifies, and writes the postmortem — escalating to a human only when risk is high.

**Repository:** `github.com/yoshithkesari/aegis` · **License:** MIT · **Status:** working closed loop, 35 tests passing.

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Problem](#2-the-problem)
3. [The Solution](#3-the-solution)
4. [What Makes AEGIS Different](#4-what-makes-aegis-different)
5. [Architecture](#5-architecture)
6. [The Incident Lifecycle](#6-the-incident-lifecycle)
7. [The Hero Technique — Label-Free Validation (CBPE)](#7-the-hero-technique--label-free-validation-cbpe)
8. [Tech Stack](#8-tech-stack)
9. [Repository Structure](#9-repository-structure)
10. [Module Reference](#10-module-reference)
11. [Installation](#11-installation)
12. [Usage](#12-usage)
13. [Configuration](#13-configuration)
14. [The Streamlit Demo](#14-the-streamlit-demo)
15. [Testing](#15-testing)
16. [Deployment](#16-deployment)
17. [Security & Privacy](#17-security--privacy)
18. [Results & Scoreboard](#18-results--scoreboard)
19. [Design Decisions](#19-design-decisions)
20. [Limitations & Future Work](#20-limitations--future-work)
21. [Glossary](#21-glossary)

---

## 1. Overview

AEGIS is an autonomous ML-reliability engineer. Production models decay silently:
as the real world shifts, a model's inputs drift (**data / covariate drift**) or
the input→output relationship changes (**concept drift**), and the model keeps
returning confident predictions while quietly getting worse. Existing tools
(Arize, WhyLabs, Fiddler) are excellent at **detecting** drift — then they stop
and page a human. The entire expensive remediation lifecycle
(investigate → decide → retrain → validate → redeploy → verify) stays manual.

AEGIS closes that loop. It runs the full lifecycle autonomously, under explicit
risk guardrails, and can validate a fix **at decision time** even when
ground-truth labels are weeks away.

| Property | Value |
|---|---|
| Running cost | **$0** — all open-source / free-tier, no credit card |
| Data egress | **None** by default (loopback binds, local reasoning option) |
| Model weights hosted | **0** (hosted LLM is optional; runs offline via heuristic) |
| Tests | **35 passing** |
| Deploy target | Streamlit Community Cloud / Hugging Face Spaces (free) |

---

## 2. The Problem

Models rot in production, and the tooling stops at the alarm. Two things make
autonomous remediation genuinely hard — and are exactly where most solutions
hand-wave:

- **Root cause is a search problem.** "Performance dropped" doesn't tell you
  *why*. One segment? A schema break? An upstream bug? True concept drift? You
  have to **investigate** — the branch can't be pre-scripted.
- **You often can't validate a retrain.** In fraud / credit / churn,
  ground-truth **labels arrive days-to-weeks late**. So a naive "retrain and
  check accuracy" gate **doesn't work at decision time**.

---

## 3. The Solution

Treat model degradation like an SRE incident. AEGIS:

1. **Detects** drift on a live prediction stream.
2. **Opens an incident** (durable, auditable).
3. **Investigates** the root cause with a scoped, read-only LLM agent.
4. **Decides** a remediation under an explicit risk matrix.
5. **Retrains** a challenger model.
6. **Validates** it **without labels** (CBPE) — the hero technique.
7. **Deploys** via canary and **promotes**, or **rolls back** / **escalates**.
8. **Records** the full audit trail and (optionally) writes a postmortem.

---

## 4. What Makes AEGIS Different

Four differentiators the incumbents don't have:

1. **Incidents & postmortems, not alerts.** AEGIS opens an incident, fixes it,
   and files a human-readable record — instantly legible.
2. **Risk-gated autonomy.** A decision matrix auto-fixes the safe cases and
   escalates the dangerous ones with a full diagnosis. Autonomy you can defend.
3. **Label-free validation.** Estimates a retrain's live performance *without
   ground truth* (CBPE), so the deploy gate works even when labels are weeks away.
4. **Provable root cause.** A drift-injection testbed with *known* causes lets
   you score diagnosis accuracy objectively.

**The one-line pitch:** *Everyone detects; only AEGIS closes the loop — and it
can validate the fix before the labels ever arrive.*

**Illustrative before / after (live in the demo):**

| Stage | Accuracy | Meaning |
|---|---|---|
| Healthy baseline | ~0.93 | model working |
| Under drift | ~0.67 | **where incumbents stop** — detect, alert, page a human |
| AEGIS recovered | ~0.92 | auto-remediated, **validated with zero labels** |

---

## 5. Architecture

### 5.1 Two loops — and only one is agentic

The core principle: **everything that can be deterministic *is***. The LLM is
scoped to exactly one job — root-cause investigation — because that's the only
part that is an open-ended search.

```
                       ┌──────────────────────────────────────────────┐
   prediction stream   │            OUTER LOOP (deterministic)         │
   ───────────────────▶│   controller · risk-gate · remediation        │
                       │   drives the incident state machine           │
                       │   executes EVERY write action                 │
                       └───────────────┬──────────────────────────────┘
                                       │ opens on incident
                                       ▼
                       ┌──────────────────────────────────────────────┐
                       │            INNER LOOP (agentic)               │
                       │   investigator + read-only tools + Reasoner   │
                       │   hypothesize → tool → observe → refine        │
                       │   READ-ONLY — can investigate, cannot deploy  │
                       └──────────────────────────────────────────────┘
```

| | Outer loop | Inner loop |
|---|---|---|
| **Role** | Drives the incident state machine; executes every write action | Root-cause investigation only |
| **LLM?** | **None** — retrain trigger & deploy gate are numeric thresholds | Yes, but **read-only** |
| **Why** | Must be predictable, idempotent, auditable — the safety-critical path | The search is data-dependent, can't be pre-scripted |

> **The LLM recommends. The controller acts.** That read/write split is a
> **security control**, not just tidiness: even a prompt-injected or
> hallucinating model physically cannot deploy anything. This is enforced by a
> test (`tests/test_read_write_split.py`) that fails if an investigation tool
> ever grows a write method.

### 5.2 Planes

- **Data plane** — stream replayer (+ drift injector) → model serving (FastAPI)
  → predictions → shadow / canary router.
- **Control plane** (deterministic) — the incident state machine, detectors,
  risk gate, remediation.
- **Orchestration** — a LangGraph state graph sequences the control-plane stages
  and branches at the real decision points (with a plain-Python fallback).
- **Stores** — model registry (MLflow), incident store (SQLite), metrics (DuckDB).

---

## 6. The Incident Lifecycle

The controller is a deterministic state machine. States:

```
HEALTHY → DRIFT_SUSPECTED → INVESTIGATING → DIAGNOSED → RETRAINING
        → VALIDATING → CANARY → PROMOTED → HEALTHY
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                 ▼
               ESCALATED       ROLLED_BACK      (log only, low severity)
              (human)          (keep champion)
```

Every transition is persisted to the incident store with a unique, sequenced
incident ID, producing a durable audit trail such as:

```
healthy         → drift_suspected   Drift detected
drift_suspected → investigating     Starting root cause investigation
investigating   → diagnosed         Investigation complete: <root cause>
diagnosed       → retraining        Starting model retraining
retraining      → validating        Label-free validation (CBPE)
validating      → canary            Canary deployment
canary          → promoted          Challenger promoted
promoted        → healthy           Incident resolved
```

### 6.1 The risk gate — autonomy as a designed feature

| Drift severity | Labels? | Label-free recovery est. | Controller action |
|---|---|---|---|
| Low | — | — | Log only |
| Medium | Yes | — | Auto-retrain → validate → canary → promote |
| Medium | Delayed | High | Auto-retrain → shadow-hold until labels confirm |
| High | any | any | Escalate + recommended fix + rollback armed |
| any | — | Would regress | Block — keep champion, alert |

---

## 7. The Hero Technique — Label-Free Validation (CBPE)

**The problem it solves:** in fraud / credit / churn, labels arrive weeks late,
so the naive "retrain then check accuracy" gate can't run at decision time.

**Confidence-Based Performance Estimation (CBPE)** estimates a model's accuracy
using only its predicted probabilities — no labels. For a calibrated probability
`p = P(y=1|x)` and threshold `t`, the predicted label is `ŷ = 1[p ≥ t]`, and the
**expected** confusion-matrix contributions of each sample are:

```
ŷ = 1 :  E[TP] += p        E[FP] += (1 − p)
ŷ = 0 :  E[TN] += (1 − p)  E[FN] += p
```

Summing over the unlabeled window yields an expected confusion matrix, from which
accuracy / precision / recall / F1 follow. AEGIS implements this directly (the
same estimator NannyML ships), so it runs on any Python version and has no heavy
dependency. It also fits an **isotonic calibrator** on reference data (CBPE
assumes calibration) and cross-checks with a **DLE** (Direct Loss Estimation)
estimate; the deploy gate requires **both to agree** before promoting.

**Honest caveat:** CBPE is only as good as calibration. Under drift, the
*champion's* calibration breaks, so its CBPE estimate can be optimistic — which
is precisely why AEGIS retrains; the challenger, calibrated on the recent window,
estimates cleanly. Verified in `tests/test_label_free.py`: on calibrated data the
estimate lands within ~2 points of the true (label-computed) accuracy.

Implementation: [`src/aegis/validation/label_free.py`](src/aegis/validation/label_free.py).

---

## 8. Tech Stack

All open-source and local-capable, plus one optional free hosted model.

| Layer | Choice | Why | Cost |
|---|---|---|---|
| Monitored model | XGBoost / scikit-learn | Trivial on purpose — it's the patient, not the product | $0 |
| Data / feature drift | Evidently (KS / chi²) | Statistical drift tests | $0 |
| Streaming / concept drift | River | ADWIN, DDM, Page-Hinkley | $0 |
| Label-free validation | **CBPE / DLE (built-in)** | Expected-confusion-matrix estimator, NannyML-compatible | $0 |
| Drift attribution | SHAP / PSI | Per-feature contribution | $0 |
| Registry / champion-challenger | MLflow (fallback: filesystem) | Versioning, promote, rollback | $0 |
| Orchestration | LangGraph (fallback: sequential) | State graph + agent sub-loop | $0* |
| Reasoning + postmortem | Gemini Flash (free tier) → Ollama → heuristic | Pluggable; zero local weights in demo | free tier |
| Serving + routing | FastAPI | Prediction endpoint + shadow/canary | $0 |
| Stores | SQLite + DuckDB | Zero-ops incident + metrics store | $0 |
| Dashboard / UI | Streamlit + Plotly | Live incident + scoreboard | $0 |

\* LangGraph is free — just don't enable **LangSmith** tracing (paid + sends
traces off-machine).

Every heavy backend (MLflow, LangGraph, DuckDB, google-generativeai) is
**optional** — the code detects its absence and falls back gracefully, so the app
runs on a lean host with only `streamlit / plotly / numpy / pandas / scikit-learn`.

---

## 9. Repository Structure

```
aegis/
├── pyproject.toml              # single source of deps + entrypoints
├── requirements.txt            # lean deploy deps (Streamlit Cloud)
├── README.md  DOCUMENTATION.md  LICENSE  .env.example
├── .streamlit/config.toml      # forces a light theme (deploy)
├── src/aegis/
│   ├── config.py               # one settings module (env / .env)
│   ├── system.py               # composition root: build & run a full incident
│   ├── data_plane/             # the "patient"
│   │   ├── replayer.py         # streams batches
│   │   ├── drift_injector.py   # known drifts (ground truth)
│   │   └── serving.py          # FastAPI: predict + canary
│   ├── control_plane/          # deterministic · write-authority
│   │   ├── controller.py       # incident state machine
│   │   ├── detectors.py        # Evidently + River
│   │   ├── monitor.py          # stream → detector → auto-incident
│   │   ├── risk_gate.py        # the decision matrix
│   │   └── remediation.py      # retrain / canary / rollback
│   ├── orchestration/          # LangGraph state machine (+ fallback runner)
│   │   └── graph.py
│   ├── agent/                  # INNER loop · read-only
│   │   ├── investigator.py     # hypothesize→tool→observe→refine
│   │   └── tools.py            # read-only investigation tools
│   ├── reasoning/              # pluggable LLM: Gemini → Ollama → heuristic
│   │   ├── base.py  gemini.py  local.py
│   ├── validation/             # label-free gate + invariance
│   │   ├── label_free.py       # CBPE / DLE estimator
│   │   ├── deploy_gate.py       # controller-facing gate adapter
│   │   └── invariance_suite.py
│   ├── stores/                 # durable: registry · incidents · metrics
│   │   ├── registry.py  incidents.py  metrics.py
│   ├── eval/                   # injected drift + scoreboard
│   │   ├── scenarios.py  scoreboard.py
│   └── ui/streamlit_app.py     # two-panel narrative + live demo + analyses
└── tests/                      # 35 tests
```

---

## 10. Module Reference

### `control_plane/controller.py` — `Controller`
Deterministic incident state machine and the sole write-authority. Key methods:
`handle_drift_detected(drift)`, `start_investigation()`, `evaluate_risk_gate()`,
`start_retraining()`, `start_validation()`, `start_canary()`,
`promote_challenger()`, `rollback()`. Persists every transition to an optional
`IncidentStore`. An `autopilot` flag (default `True`) makes stages self-chain;
set `False` to let the orchestrator drive stage-by-stage.

### `control_plane/risk_gate.py` — `RiskGate`
Pure decision logic mapping (drift severity, label availability, label-free
estimate, would-regress) → `Action` (`LOG_ONLY`, `AUTO_RETRAIN`,
`AUTO_RETRAIN_SHADOW_HOLD`, `ESCALATE`, `BLOCK`).

### `control_plane/detectors.py` — `DetectorSuite`
`EvidentlyDetector` (batch KS / chi² feature drift), `RiverDetector` (streaming
ADWIN/DDM/PageHinkley), `ConceptDriftDetector` (performance-based).

### `control_plane/monitor.py` — `StreamMonitor`
Watches a batch stream, runs the detector against a reference distribution, and
**auto-opens an incident** when drift crosses the bar. Guards against false
interventions with a **Bonferroni-corrected** threshold (0.05 / n_features) and a
**minimum-severity** gate; refreshes its reference after a successful remediation.

### `agent/investigator.py` — `Investigator`
The agentic inner loop. Builds context from the incident, runs read-only tools,
and calls a `Reasoner` to synthesise a diagnosis. Falls back to a deterministic
rule-based diagnosis when no reasoner is reachable.

### `agent/tools.py` — `InvestigationToolkit`
Read-only tools: `SlicePerformance`, `AttributeDrift` (PSI), `QueryDeployLog`,
`DiffSchema`, `EstimatePerfWithoutLabels`. **None can write / deploy.**

### `reasoning/` — the pluggable LLM
`Reasoner` interface (`complete(prompt) -> str`) with `GeminiReasoner` (hosted,
capped, fail-safe), `OllamaReasoner` (local, no egress), `HeuristicReasoner`
(no LLM, always available). `build_reasoner()` returns the
Gemini → Ollama → heuristic fallback chain.

### `validation/label_free.py` — `LabelFreeValidator`
The CBPE / DLE estimator (see §7). `validation/deploy_gate.py` — `DeployGate`
adapts it to the controller's `validate(incident, label_free)` interface and
**fails closed** when there's no challenger evidence.

### `stores/`
`ModelRegistry` (MLflow or filesystem fallback; champion/challenger with instant
rollback), `IncidentStore` (SQLite, durable audit trail), `MetricsStore` (DuckDB
or in-memory).

### `orchestration/graph.py`
`build_incident_graph(controller)` → a real compiled LangGraph `StateGraph` (or a
branching sequential runner) that drives the stages and branches at risk-gate /
validation / canary. `run_incident_via_graph(controller, drift)` opens and drives
an incident with autopilot off.

### `system.py` — composition root
`build_system(...)` wires every real component; `run_incident(...)` /
`run_stream(...)` drive one incident; `demo_run()` returns a full UI payload with
true before/after accuracies. `python -m aegis.system` prints a full run.

---

## 11. Installation

```bash
git clone https://github.com/yoshithkesari/aegis.git
cd aegis

# editable install with extras: [hosted] Gemini, [dev] tests
pip install -e ".[hosted,dev]"

# environment
cp .env.example .env
# edit .env and add GEMINI_API_KEY (optional — runs offline without it)
```

**Requirements:** Python ≥ 3.9. The lean runtime needs only
`streamlit, plotly, numpy, pandas, scikit-learn`; the full stack adds
`mlflow, langgraph, duckdb, evidently, river, shap, fastapi, google-generativeai`.

---

## 12. Usage

### Run the Streamlit UI (recommended)
```bash
streamlit run src/aegis/ui/streamlit_app.py
# or, after install:
aegis-ui
```

### Run one full incident on real components (CLI)
```bash
python -m aegis.system
# or:
aegis-run
```
Prints the diagnosis, deploy-gate decision, resolved state, registry version,
and the persisted audit trail.

### Run the FastAPI model server
```bash
python -m aegis.data_plane.serving
# or:
aegis-server
```

### Drive an incident from Python
```python
from aegis.system import build_system, run_stream, demo_run

system = build_system()               # real champion, stores, gate, detector
summary = run_stream(system, drifted=True)   # stream → auto-incident → resolve
print(summary["final_state"])         # 'healthy'

payload = demo_run()                  # full before/after payload for a UI
```

### Drive via the LangGraph orchestrator
```python
from aegis.system import build_system
from aegis.orchestration import run_incident_via_graph

system = build_system()
status = run_incident_via_graph(
    system.controller,
    {"drift_type": "covariate", "severity": "medium", "summary": "x"},
)
```

---

## 13. Configuration

All settings live in [`src/aegis/config.py`](src/aegis/config.py) (`Settings`),
populated from environment variables (optionally a `.env` file):

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Hosted reasoner key (optional) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Hosted model id |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | `127.0.0.1:11434` / `llama3.2` | Local reasoner |
| `MAX_LLM_CALLS` | `20` | Per-incident cap (stays in free tier) |
| `MAX_RETRAINS_PER_DAY` | `5` | Bounded-loop safety |
| `RETRAIN_COOLDOWN_SECONDS` | `900` | Adversarial-drift defense |
| `DRIFT_THRESHOLD` | `0.05` | Detection threshold |
| `REGRESSION_THRESHOLD` | `-0.05` | Deploy-gate regression bound |
| `INCIDENTS_DB` / `METRICS_DB` / `MLFLOW_TRACKING_URI` | `artifacts/…` | Store locations |
| `AEGIS_HOST` / `AEGIS_PORT` | `127.0.0.1` / `8000` | Serving bind (loopback = no egress) |
| `REDACT_BEFORE_HOSTED_LLM` | `true` | Send aggregated stats only |

---

## 14. The Streamlit Demo

The UI has a **view selector** at the top:

- **🔴 Live System (real)** — a two-panel BEFORE/AFTER narrative plus a live
  incident driven by `aegis.system`. Every number is computed live; the
  scoreboard is measured, not asserted. Includes the live "Hero Technique" panel
  (CBPE estimate vs withheld truth).
- **📊 Analysis · Model Monitoring 1** — a fixed, saved analysis
  (`fraud-classifier@prod`, covariate drift, 0.94 → 0.71 → 0.905).
- **📊 Analysis · Model Monitoring 2** — a fixed, saved analysis
  (`credit-default@prod`, concept drift, 0.887 → 0.623 → 0.861).

The two analyses are **hardcoded presets**, so their figures are byte-identical
after a reset and on every machine.

---

## 15. Testing

```bash
pip install -e ".[dev]"
pytest            # 35 tests
```

| Suite | Covers |
|---|---|
| `test_risk_gate.py` (8) | The decision matrix — pure logic, every branch |
| `test_controller.py` (4) | State-machine transitions, idempotency, escalation |
| `test_read_write_split.py` (3) | Security invariant: investigation tools have no write methods |
| `test_label_free.py` (7) | CBPE correctness properties (p=0.5→0.5, within 2pts of truth, regressive blocked) |
| `test_deploy_gate_integration.py` (3) | Controller + real gate promotes strong / blocks regressive challenger (no labels) |
| `test_monitor.py` (3) | Healthy stream opens 0 incidents; drift auto-opens 1 & resolves |
| `test_orchestration.py` (4) | LangGraph drives stages, branches, restores autopilot |
| `test_system.py` (3) | Full loop closes to healthy with a persisted audit trail; dramatic real before/after |

---

## 16. Deployment

Free, no credit card.

### Streamlit Community Cloud (recommended)
1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
2. **Create app → Deploy from GitHub.**
3. Repository `yoshithkesari/aegis`, branch `main`, **main file path**
   `src/aegis/ui/streamlit_app.py`.
4. *(Optional)* **Advanced → Secrets:** `GEMINI_API_KEY = "…"`.
5. **Deploy.** First build ~1–2 min; you get a public `*.streamlit.app` URL.

### Hugging Face Spaces (16 GB RAM)
Create a **Streamlit** Space, connect the repo, set the app file to
`src/aegis/ui/streamlit_app.py`, add `GEMINI_API_KEY` in Settings → Secrets.

**Notes**
- `requirements.txt` is lean on purpose; uncomment the optional backends there
  to run real MLflow/LangGraph/DuckDB in the cloud.
- `.streamlit/config.toml` pins a light theme so text is dark-on-white regardless
  of the viewer's default (fixes washed-out text).
- Free apps sleep when idle — **warm the URL** a few minutes before presenting.
- Persistence (SQLite / MLflow dirs) writes to ephemeral container disk; point
  `INCIDENTS_DB` / `MLFLOW_TRACKING_URI` at durable storage for production.

---

## 17. Security & Privacy

**Guaranteed no charges**
- Attach no credit card anywhere — no card, no possible charge.
- Free-tier LLM keys rate-limit (`429`), never bill, if no billing is linked.
- Bounded loops: max tool-calls/incident, max retrains/day, cooldown, kill-switch.

**No data leakage**
- Local reasoner = no egress; hosted fallback sends **aggregated stats only**.
- Bind MLflow / Streamlit / FastAPI to `127.0.0.1`.
- Demo uses **synthetic data** — no real PII exists to leak.

**Deployment security**
- **Read/write separation** — the LLM can investigate, never deploy (enforced by test).
- **Data-poisoning defense** — the risk gate blocks retrains that would regress.
- **Canary + auto-rollback** — blast radius is bounded; previous champion retained.
- **Full audit trail** — every controller action is persisted.

**Adversarial-drift threat.** An attacker could deliberately inject drift to trick
the agent into retraining on poisoned data. Defenses: the label-free estimate
blocks regressive retrains, a retrain-frequency cap + cooldown apply, the previous
champion is always kept for instant rollback, and suspicious patterns escalate to
a human instead of auto-fixing.

---

## 18. Results & Scoreboard

Because AEGIS **injects** the drift in the eval harness, it owns the ground truth
— turning "trust us" into a measured scoreboard. Representative live results:

| Metric | Value | Meaning |
|---|---|---|
| Before (champion under drift) | ~0.67 | where incumbents stop |
| After (AEGIS recovered) | ~0.92 | autonomously restored |
| False interventions (healthy stream) | **0** | doesn't retrain on noise |
| Incidents auto-resolved | 1 per drift | loop closes |
| Deploy gate decision | PROMOTE | validated with **no labels** |
| Audit-trail steps persisted | 8 | full lifecycle recorded |

Saved analyses (fixed): Analysis 1 — MTTR 38s, RCA top-1 10/12, RCA 83%;
Analysis 2 — MTTR 51s, RCA top-1 13/15, RCA 87%.

---

## 19. Design Decisions

- **Deterministic controller, agentic investigator.** The safety-critical write
  path is LLM-free and auditable; the LLM is boxed to the one open-ended task.
- **Read/write split as a security boundary**, verified by an automated test.
- **CBPE implemented directly**, not via NannyML, to run on any Python and stay
  dependency-light; both CBPE and DLE must agree before a promotion.
- **Graceful degradation everywhere** — MLflow / LangGraph / DuckDB / Gemini are
  optional; the system runs fully on a minimal host with fallbacks.
- **Two orchestration modes** — the controller can self-drive (autopilot) or be
  sequenced by a LangGraph state graph; both reach identical outcomes.
- **Bonferroni + min-severity gating** on the stream monitor to guarantee zero
  false interventions on same-distribution data.

---

## 20. Limitations & Future Work

- **Label-free estimate depends on calibration.** Under heavy drift the champion's
  own estimate can be optimistic; mitigated by retraining + the both-must-agree
  gate, but worth surfacing.
- **NannyML not used** (self-implemented CBPE); wiring real NannyML would require
  a Python < 3.13 environment.
- **Pure concept drift** (label relationship changes with unchanged feature
  distributions) is not caught by the feature-drift monitor alone — the
  performance-based `ConceptDriftDetector` covers this path and is a good area to
  deepen.
- **Ephemeral persistence** on free hosts; production would point stores at
  durable storage.
- **Future:** Slack/PagerDuty incident cards, a second real dataset to show
  generality, and LLM-authored postmortems on every incident.

---

## 21. Glossary

| Term | Meaning |
|---|---|
| **Covariate / data drift** | The input distribution changes |
| **Concept drift** | The input→output relationship changes |
| **Champion / challenger** | The live model vs a candidate replacement |
| **CBPE** | Confidence-Based Performance Estimation — accuracy estimate without labels |
| **DLE** | Direct Loss Estimation — a complementary label-free estimate |
| **Canary** | A limited-traffic deployment to bound blast radius |
| **MTTR** | Mean Time To Remediate |
| **Risk gate** | The decision matrix that gates autonomous actions |
| **Autopilot** | Controller mode where stages self-chain vs graph-driven |

---

*AEGIS — Autonomous Model Reliability. Zero cost, zero data egress. The LLM can
only investigate, never deploy; a deterministic controller executes remediation
under risk guardrails with canary and auto-rollback.*
