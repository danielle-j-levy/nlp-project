import os

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

DATASET = "allenai/WildChat-1M"
N_SHARDS = 14
SHARD_FMT = "datasets/{repo}/data/train-{i:05d}-of-{n:05d}.parquet"


def load_hf_token(env_path=None):
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env_path = env_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line.startswith("HF_TOKEN") and "=" in line:
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["HF_TOKEN"] = tok
                return tok
    return None


def _first_user_text(conversation):
    for turn in conversation:
        if turn.get("role") == "user":
            return turn.get("content") or ""
    return conversation[0].get("content", "") if conversation else ""


def iter_first_turns(max_rows=None, shards=None, fs=None):
    """Yield dict(conversation_hash, timestamp, text) = first user turn per
    conversation, streamed row-group by row-group (no full-dataset RAM)."""
    load_hf_token()
    fs = fs or HfFileSystem()
    shard_idxs = shards if shards is not None else range(N_SHARDS)
    cols = ["conversation_hash", "timestamp", "conversation"]
    yielded = 0
    for si in shard_idxs:
        path = SHARD_FMT.format(repo=DATASET, i=si, n=N_SHARDS)
        with fs.open(path) as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=cols)
                chash = tbl.column("conversation_hash").to_pylist()
                ts = tbl.column("timestamp").to_pylist()
                conv = tbl.column("conversation").to_pylist()
                for h, t, c in zip(chash, ts, conv):
                    text = _first_user_text(c)
                    if not text:
                        continue
                    yield {"conversation_hash": h, "timestamp": t, "text": text}
                    yielded += 1
                    if max_rows and yielded >= max_rows:
                        return
