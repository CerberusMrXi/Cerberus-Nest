# 🧠 CERBERUS NEST

### Developmental Artificial Intelligence Research Platform

> **An experimental digital infant for studying developmental learning, memory, self-models, world models, and consciousness-related behavior.**

**Author:** Sudeepa Wanigarathna  
**Version:** `0.1.0` — Baseline Prototype  
**Status:** 🚧 Active Research / Experimental

---

## 🧬 What is CERBERUS NEST?

CERBERUS NEST is an experimental artificial-life/developmental-AI project.

The long-term research question is:

> **How much developmental intelligence can emerge when an artificial agent begins with limited knowledge and learns through embodied interaction with an environment?**

The current prototype is a **2D research environment**, not a claim of artificial consciousness.

The baseline already contains:

- 👁️ CNN-based visual encoding
- 🧠 reinforcement learning
- 🔎 intrinsic curiosity / RND
- 💾 episodic memory
- 🧩 semantic memory
- ⚙️ procedural memory
- 🎯 attention / global-workspace-inspired selection
- 🌎 world-model prediction
- 🪞 self-model prediction
- 🗣️ experimental language association
- 🧒 developmental stages
- ❤️ internal drives such as hunger, thirst, fatigue, curiosity and social drive
- 📊 CSV experiment logging

---

## 🏗️ Current Architecture

```text
                 DIGITAL AGENT
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      Vision        Internal       Previous
      Input           State          Action
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                  Representation
                       │
              ┌────────┴────────┐
              ▼                 ▼
          Attention          Memory
              │                 │
              └────────┬────────┘
                       ▼
                 Decision Core
                       │
                       ▼
                     Action
                       │
                       ▼
                 2D Environment
                       │
                       ▼
                    Reward
                       │
                       └──────► Learning
```

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/CerberusMrXi/Cerberus-Nest.git
cd Cerberus-Nest
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install

```bash
pip install -r requirements.txt
```

### 4. Run the baseline

```bash
python -m cerberus_nest
```

or:

```bash
cerberus-nest
```

The baseline runs headless and writes experiment metrics to:

```text
logs/baseline_42.csv
```

### 5. Run the original module directly

```bash
python src/cerberus_nest/core.py
```

---

## 🔬 Research Roadmap

### v0.1 — Baseline Prototype
- [x] 2D world
- [x] internal drives
- [x] visual encoder
- [x] Q-learning
- [x] curiosity
- [x] memory modules
- [x] self/world prediction
- [x] attention mechanism
- [x] developmental stages
- [x] experiment logging

### v0.2 — Real Developmental Learning
- [ ] separate observation representation from semantic object labels
- [ ] improved replay and learning stability
- [ ] persistent agent identity across episodes
- [ ] real memory retrieval
- [ ] learned attention instead of heuristic attention
- [ ] proper self-model evaluation
- [ ] proper metacognition metrics
- [ ] checkpointing
- [ ] experiment configuration system

### v0.3 — Embodied Development
- [ ] richer body model
- [ ] object permanence experiments
- [ ] spatial memory
- [ ] sensory prediction
- [ ] action ownership
- [ ] temporal continuity
- [ ] sleep/memory consolidation

### v0.4 — Social Development
- [ ] caregiver agent
- [ ] imitation
- [ ] joint attention
- [ ] social memory
- [ ] interaction learning

### v0.5 — Communication
- [ ] vocalization model
- [ ] symbol grounding experiments
- [ ] emergent communication
- [ ] caregiver-agent language environment

### v1.0 — Developmental Research Platform
- [ ] reproducible experiment suite
- [ ] multi-agent experiments
- [ ] ablation studies
- [ ] experiment dashboard
- [ ] life replay
- [ ] long-term developmental evaluation

---

## ⚠️ Important Scientific Position

CERBERUS NEST **does not claim to create or detect consciousness**.

Metrics such as self-model accuracy, temporal continuity, metacognitive accuracy, or body-ownership behavior are **experimental behavioral measurements**.

A high score must not automatically be interpreted as consciousness.

The project should distinguish:

```text
OBSERVATION
    ↓
MEASUREMENT
    ↓
INTERPRETATION
```

and document alternative explanations.

---

## 📁 Repository Structure

```text
cerberus-nest/
├── src/
│   └── cerberus_nest/
│       ├── __init__.py
│       ├── __main__.py
│       └── core.py
├── configs/
│   └── baseline.yaml
├── docs/
│   ├── architecture.md
│   ├── research.md
│   └── experiments.md
├── scripts/
├── tests/
├── logs/
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 🧪 Research Philosophy

The system should gradually move away from hand-designed semantic assumptions.

Instead of:

```text
"this is food"
```

the eventual system should receive sensory observations and discover useful regularities through experience.

The research direction is:

```text
Experience
    ↓
Perception
    ↓
Prediction
    ↓
Action
    ↓
Consequence
    ↓
Memory
    ↓
Learning
    ↓
Adaptation
```

---

## 👤 Author

**Sudeepa Wanigarathna**

CERBERUS NEST is an independent experimental research project exploring developmental AI, artificial life, machine learning, and consciousness-related computational models.

---

## 📜 License

MIT License.
