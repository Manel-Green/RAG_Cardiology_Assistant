"""
Step 3: Generate embeddings for every chunk and prepare the files
Vertex AI Vector Search needs to build an index.

Vector Search only stores IDs and vectors - it doesn't store the actual
chunk text. So this script writes two things:

1. embeddings.jsonl  - {"id": ..., "embedding": [...]}  one line per
   chunk, in the exact format Vector Search needs for index creation.
2. chunk_lookup.json - {id: {"text": ..., "source_file": ...}} so that
   once Vector Search returns a matching ID, we can look up what text
   it actually corresponds to.

Reads from:  gs://manel-portfolio-demos-rag-heart-studies/processed/chunks/
Writes to:   gs://manel-portfolio-demos-rag-heart-studies/processed/embeddings/

Run: python src/build_embeddings.py
"""

import json

import numpy as np
import vertexai
from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel

PROJECT_ID = "manel-portfolio-demos"
VERTEX_LOCATION = "us-central1"
BUCKET = "manel-portfolio-demos-rag-heart-studies"
INPUT_PREFIX = "processed/chunks/"
OUTPUT_PREFIX = "processed/embeddings/"

# Set to a source file name (e.g. "Myocardial_Viability.json") to test on
# just that one document's chunks before running on all 6.
ONLY_SOURCE = None

vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")


def estimate_tokens(text):
    # Rough heuristic (~4 characters per token for English text) - no
    # exact tokenizer available here, so this stays conservative via
    # MAX_BATCH_TOKENS leaving headroom below the real 20,000 limit.
    return len(text) // 4 + 1


def embed_texts(texts, max_batch_size=250, max_batch_tokens=15000):
    # The API caps both: 250 instances per batch AND ~20,000 tokens per
    # batch. Chunk sizes vary a lot (title lines vs. near-400-word
    # chunks), so a fixed instance-count batch can still blow the token
    # cap - flush whenever either limit would be exceeded.
    all_embeddings = []
    batch = []
    batch_tokens = 0

    for text in texts:
        text_tokens = estimate_tokens(text)
        if batch and (
            len(batch) >= max_batch_size or batch_tokens + text_tokens > max_batch_tokens
        ):
            embeddings = embedding_model.get_embeddings(batch)
            all_embeddings.extend(np.array(e.values) for e in embeddings)
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += text_tokens

    if batch:
        embeddings = embedding_model.get_embeddings(batch)
        all_embeddings.extend(np.array(e.values) for e in embeddings)

    return all_embeddings


def build_embeddings():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)

    blobs = [b for b in bucket.list_blobs(prefix=INPUT_PREFIX) if b.name.endswith(".json")]

    if ONLY_SOURCE:
        blobs = [b for b in blobs if b.name.endswith(ONLY_SOURCE)]
        print(f"ONLY_SOURCE is set - processing just '{ONLY_SOURCE}' for this test run.")

    print(f"Found {len(blobs)} chunk file(s) to embed.")

    all_ids = []
    all_texts = []
    chunk_lookup = {}

    for blob in blobs:
        data = json.loads(blob.download_as_text())
        source_file = data["source_file"]
        for chunk in data["chunks"]:
            chunk_id = f"{source_file}__{chunk['chunk_id']}"
            all_ids.append(chunk_id)
            all_texts.append(chunk["text"])
            chunk_lookup[chunk_id] = {
                "text": chunk["text"],
                "source_file": source_file,
            }

    print(f"Embedding {len(all_texts)} chunks...")
    embeddings = embed_texts(all_texts)

    embeddings_lines = []
    for chunk_id, embedding in zip(all_ids, embeddings):
        embeddings_lines.append(
            json.dumps({"id": chunk_id, "embedding": embedding.tolist()})
        )

    # Lives in its own subfolder, separate from chunk_lookup.json below -
    # Vector Search index creation reads every file in the folder you
    # point it at and expects each one to be in this exact format.
    # Extension must be .json (not .jsonl) - Vector Search only
    # recognizes .json/.csv/.avro, even though the content here is
    # JSON-lines format (one JSON object per line).
    embeddings_blob = bucket.blob(f"{OUTPUT_PREFIX}vectors/embeddings.json")
    embeddings_blob.upload_from_string(
        "\n".join(embeddings_lines), content_type="application/json"
    )
    print(f"-> wrote {len(embeddings_lines)} vectors to gs://{BUCKET}/{OUTPUT_PREFIX}vectors/embeddings.json")

    lookup_blob = bucket.blob(f"{OUTPUT_PREFIX}chunk_lookup.json")
    lookup_blob.upload_from_string(
        json.dumps(chunk_lookup, indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    print(f"-> wrote lookup file to gs://{BUCKET}/{OUTPUT_PREFIX}chunk_lookup.json")


if __name__ == "__main__":
    build_embeddings()
