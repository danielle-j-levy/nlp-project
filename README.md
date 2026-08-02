# Reichman University NLP Project

Measures cross-request token reuse across a range of LLM workloads (chat, coding
agents, web agents, operational telemetry, stateful APIs, retrieval, and SQL
schemas). Each workload has its own script in `experiments/`; the shared
measurement code lives in `core/`.

## Run

```bash
pip install -r requirements.txt
export HF_TOKEN=...                 # needed for the HuggingFace-hosted corpora
python experiments/bgl_syslog.py    # any workload
```

Or open `results.ipynb` — it is fully self-contained (every function inlined; the
only imports are the standard library plus `transformers`/`datasets`/
`huggingface_hub`/`pyarrow`/`pandas`, auto-installed by the first cell). It needs
no other file from this repo, so it runs as a standalone upload in Google Colab.
Set `HF_TOKEN` for the HuggingFace-hosted corpora; local-data corpora read from
`./data/…`.

Each script reports prefix caching, PIC, and PIC-processed reuse (with its
in-session / cross-session split). The accounting rules — second-occurrence
reuse, whole-stream denominator, one trajectory per task, ≥500-token spans, and
prefix-cache-disjoint credit — live in `core/prefix_disjoint.py` and
`core/analyze_rules.py`.

## Data

Corpora are large and are not committed. HuggingFace-hosted ones download on
first run; the rest are read from local `data/` — see `data/README.md` for
sources and expected paths.
