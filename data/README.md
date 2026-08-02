# Data sources

Large corpora are not committed. Scripts run from the repo root and read local
inputs from this `data/` directory; HuggingFace-hosted corpora download on first
run (set `HF_TOKEN`).

## Downloads automatically (HuggingFace)

| Rows | Dataset |
|------|---------|
| 01 | `allenai/WildChat-1M` |
| 02 | `google-research-datasets/paws` |
| 03 | `SWE-bench/SWE-smith-trajectories` |
| 04 | `nebius/SWE-rebench-openhands-trajectories` |
| 05 | `zai-org/CC-Bench-trajectories` |
| 06 | `nebius/SWE-agent-trajectories` |
| 07 | `stanfordnlp/nnetnav-wa` |
| 08 | `osunlp/Mind2Web` |
| 09 | `stanfordnlp/nnetnav-live` |
| 10 | `ibm-research/ITBench-Trajectories` |
| 11 | `ibm-research/ITBench-Lite` (snapshots/ciso) |
| 12 | ITBench alerts (Scenario-*/alerts) |
| 22 | `birdsql/bird_sql_dev_20251106` + `target-benchmark/bird-corpus-validation` |

## Read from local `data/` (place before running)

| Row | Expected path |
|-----|---------------|
| 13 | `data/loghub/BGL_200k.log` (LogHub BGL) |
| 14 | `data/appworld/hf_traces/halo_gemini3flash_traces.jsonl` |
| 15–17 | `data/tau2/final/*.json` (sierra-research/tau2-bench results) |
| 18 | `data/mtrag/conversations/conversations_human.json` |
| 19 | `data/multidoc2dial/multidoc2dial_{doc,dial_train,dial_validation}.json` |
| 20 | `data/tauknowledge/banking_knowledge/{documents/*.json,tasks.json}` |
| 21 | `data/spider2/` (spider2-lite.jsonl, databases/**/DDL.csv, documents/) |
| 23 | `data/livesqlbench/large-v1/` |
| 24 | `data/birdinteract/{lite,full}/**/bird_interact_data.jsonl` |
