# Federated Research Agent

An agentic research assistant that answers biomedical research questions across multiple data sites **without moving patient-level data**. The agent plans the analysis, queries each site through privacy-preserving tools, combines the results by federated meta-analysis and asks a human to approve high-stakes steps.

> 🚧 Work in progress. All models run locally (Ollama), and all data is synthetic.

## Why this design

- **Compute goes to the data.** Each site (a "node") keeps its own database and releases only aggregates that pass its disclosure-control policy. The design follows the secure-processing-environment model of the European Health Data Space (EHDS).
- **No SQL reaches the nodes.** Clients send typed cohort and analysis specifications drawn from a closed vocabulary (SNOMED CT, LOINC, RxNorm). Each node compiles them to parameterised SQL. A prompt-injected agent therefore has no way to ask a node for raw rows.
- **Every release is audited.** Each node keeps an append-only log of every request it accepted, suppressed or rejected, in the style of a data airlock.
- **Ground truth by construction.** The synthetic sites contain planted effects, including a deliberate treatment-confounding trap. Any conclusion the agent reaches can therefore be scored automatically against the true answer.

## Status

- [x] Synthetic multi-site data (simplified OMOP CDM-inspired schema) with documented generative model
- [x] Federated node API: cohort/summary/regression, per-node small-cell suppression and rounding, audit log
- [x] Federated meta-analysis (fixed and DerSimonian–Laird random effects), tested against a centralised analysis
- [ ] LangGraph agent (plan-and-execute, tool use, human-in-the-loop approval)
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

API docs for each node are served at `http://127.0.0.1:8101/docs` (and on ports 8102 and 8103).
