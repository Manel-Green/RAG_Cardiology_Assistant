"""
Diagnostic: check embeddings.jsonl for structural problems before
retrying index creation again - malformed JSON lines, inconsistent
embedding dimensions, duplicate IDs, or NaN/Infinity values. Runs
locally against the file already in GCS, no index-build cost involved.

Run: python src/validate_embeddings.py
"""

import json
import math

from google.cloud import storage

PROJECT_ID = "manel-portfolio-demos"
BUCKET = "manel-portfolio-demos-rag-heart-studies"
BLOB_PATH = "processed/embeddings/vectors/embeddings.jsonl"


def validate():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(BLOB_PATH)
    content = blob.download_as_text()

    lines = content.strip().split("\n")
    print(f"Total lines: {len(lines)}")

    ids_seen = set()
    dims = set()
    bad_lines = []

    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            bad_lines.append((i, f"invalid JSON: {e}"))
            continue

        if "id" not in data or "embedding" not in data:
            bad_lines.append((i, "missing id or embedding field"))
            continue

        id_ = data["id"]
        if id_ in ids_seen:
            bad_lines.append((i, f"duplicate id: {id_}"))
        ids_seen.add(id_)

        emb = data["embedding"]
        dims.add(len(emb))

        if any(math.isnan(x) or math.isinf(x) for x in emb):
            bad_lines.append((i, f"NaN/Inf in embedding for id {id_}"))

        if not id_ or len(id_.encode("utf-8")) > 128:
            bad_lines.append((i, f"id empty or over 128 bytes: {id_}"))

    print(f"Unique IDs: {len(ids_seen)}")
    print(f"Embedding dimensions seen: {dims}")
    print(f"Problem lines: {len(bad_lines)}")
    for i, reason in bad_lines[:20]:
        print(f"  line {i}: {reason}")


if __name__ == "__main__":
    validate()
