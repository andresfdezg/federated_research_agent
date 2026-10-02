# Federated Research Agent

An agentic research assistant that answers biomedical research questions across multiple data sites **without moving patient-level data**. The agent plans the analysis, queries each site through privacy-preserving tools, combines the results by federated meta-analysis and asks a human to approve high-stakes steps.

> 🚧 Work in progress. All models run locally (Ollama), and all data is synthetic.

## Why this design

- **Compute goes to the data.** Each site (a "node") keeps its own database and releases only aggregates that pass its disclosure-control policy. The design follows the secure-processing-environment model of the European Health Data Space (EHDS).
- **No SQL reaches the nodes.** Clients send typed cohort and analysis specifications drawn from a closed vocabulary (SNOMED CT, LOINC, RxNorm). Each node compiles them to parameterised SQL. A prompt-injected agent therefore has no way to ask a node for raw rows.
- **Every release is audited.** Each node keeps an append-only log of every request it accepted, suppressed or rejected, in the style of a data airlock.
- **Ground truth by construction.** The synthetic sites contain planted effects, including a deliberate treatment-confounding trap. Any conclusion the agent reaches can therefore be scored automatically against the true answer.

## How the agent works

```
discover nodes → plan (structured output) → human approval ─→ execute on every node → reflect → report
                   ↑   ↳ validator: errors fed back   │ revise          │ meta-analysis      │ needs more
                   └──────────────────────────────────┴─────────────────────────────────────┘
```

- **Plans are typed, not free text.** The local LLM (qwen3:8b via Ollama) emits a JSON-schema-constrained plan, and a validator rejects analyses that cannot work, such as a covariate made constant by the cohort definition. The validator's explanation goes back to the model, which then corrects the plan.
- **Humans approve every query before it reaches the data.** LangGraph `interrupt()` pauses the graph so a reviewer can approve, revise with feedback or reject the plan. This implements the human oversight required by the EU AI Act (Art. 14).
- **The LLM never does arithmetic.** Pooling, meta-analysis, odds ratios and CIs are computed by tested code; the model only interprets them.

### Example: the reviewer catches a confounder

*"Is LDL cholesterol associated with coronary artery disease?"* The agent's first plan adjusts only for age and sex and finds **OR 0.99 per mg/dL**, an apparently *protective* effect. That result is an artefact: patients start statins after a CAD diagnosis, and the statins lower their measured LDL. When a reviewer flags this at the approval step, the agent replans with statin use as a covariate and recovers the true positive association, **OR 1.006 per mg/dL (95% CI 1.004–1.008), I² = 0%**, consistent across the three nodes.

## Status

- [x] Synthetic multi-site data (simplified OMOP CDM-inspired schema) with documented generative model
- [x] Federated node API: cohort/summary/regression, per-node small-cell suppression and rounding, audit log
- [x] Federated meta-analysis (fixed and DerSimonian–Laird random effects), tested against a centralised analysis
- [x] LangGraph agent: plan → human approval → federated execution → reflection/replanning → report
- [x] Plan validator that catches impossible analyses and feeds errors back to the LLM for self-correction
- [ ] TypeScript UI
- [ ] Evaluation: LLM-as-judge, ground-truth scoring, red-teaming (prompt injection, differencing attacks)
- [ ] Literature RAG (PubMed) and sandboxed code execution
- [ ] LoRA fine-tuning for function calling; Kubernetes deployment

## Quickstart

```bash
uv sync
uv run python -m fedresearch.data.generate   # writes data/nodes/*.duckdb
uv run python -m fedresearch.node            # starts the 3 nodes on ports 8101-8103
uv run pytest
```

Then, with [Ollama](https://ollama.com) running and `ollama pull qwen3:8b` done:

```bash
uv run python -m fedresearch.agent "Do BRCA1 carriers get breast cancer at a younger age than non-carriers?"
```

API docs for each node are served at `http://127.0.0.1:8101/docs` (and on ports 8102 and 8103).
