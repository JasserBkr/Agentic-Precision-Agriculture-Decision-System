# Project 12 — Agentic Precision-Agriculture Decision System

DeepShift AI Summer Internship 2026. An agentic system fusing Sentinel-2
satellite imagery, Open-Meteo weather/soil forecasts, and IoT soil-moisture
streams into defensible, explained irrigation and fertilization
recommendations, using a LangGraph ReAct agent on top of a deterministic
data-and-forecasting pipeline (Chronos-2 zero-shot + TFT/N-HiTS baseline).

See `docs/sota_note.pdf` for the full architectural justification.

## Setup

1. Install [`uv`](https://docs.astral.sh/uv/) if you haven't already.
2. Clone this repo and run:
   ```bash
   uv sync
   ```
3. Copy `.env.example` to `.env` and fill in your Copernicus and Earth Engine
   credentials (see `docs/` or ask a teammate if unsure how to generate
   these).
4. Authenticate Earth Engine once locally:
   ```bash
   uv run python -c "import ee; ee.Authenticate()"
   ```
5. Edit `configs/field.yaml` with your sample field's bounding box.

## Project structure

```
src/agri_agent/
├── utils/          # auth, logging — shared across everything
├── data_access/     # Week 1–2: satellite, weather, IoT retrieval + fusion
├── forecasting/      # Week 3–4: Chronos-2, TFT/N-HiTS baseline, evaluation
├── agent/            # Week 5–6: LangGraph state, tools, graph, prompts
└── dashboard/        # Week 7: Streamlit + Folium interface
```

## Running the pipeline

```bash
uv run python scripts/run_pipeline.py
```

To ask your own questions in an interactive loop (type `exit`/`quit` or Ctrl-D
to leave):

```bash
uv run python scripts/run_pipeline.py --interactive
```

## Running tests

```bash
uv run pytest
```
