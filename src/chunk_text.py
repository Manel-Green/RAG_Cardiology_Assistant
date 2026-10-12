"""
Step 2b: Hybrid semantic chunking.

Document AI already split each page into paragraphs when it ran in
extract_text.py - we use those as the base unit so a chunk never cuts
a sentence in half. Each paragraph gets embedded with Vertex AI's
embedding model, then we walk through paragraphs in order: while
consecutive paragraphs stay semantically similar, they merge into the
same chunk (same topic continuing); when similarity drops (a topic
shift) or a chunk hits the size cap, a new chunk starts.

Reads from:  gs://manel-portfolio-demos-rag-heart-studies/processed/docai-raw/
Writes to:   gs://manel-portfolio-demos-rag-heart-studies/processed/chunks/

Run: python src/chunk_text.py
"""

import json
import re
from collections import defaultdict

import numpy as np
import vertexai
from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel

PROJECT_ID = "manel-portfolio-demos"
VERTEX_LOCATION = "us-central1"
BUCKET = "manel-portfolio-demos-rag-heart-studies"
INPUT_PREFIX = "processed/docai-raw/"
OUTPUT_PREFIX = "processed/chunks/"

# The similarity threshold is no longer a fixed number (that didn't
# generalize - see BREAKPOINT_PERCENTILE below). MAX_CHUNK_WORDS is
# still a hard cap regardless of similarity.
MAX_CHUNK_WORDS = 400

# Adaptive breakpoint: for each document, look at the similarity between
# every consecutive paragraph pair, and only start a new chunk where the
# drop is unusually large *for that document* - specifically, in the
# bottom 20% of transitions. This calibrates per-document instead of
# comparing every paper against one arbitrary global number.
BREAKPOINT_PERCENTILE = 20

# Set to a source document name (e.g. "Myocardial_Viability") to test the
# chunking logic on just that one document before running against all 6.
# Set to None to process everything.
ONLY_SOURCE = None

vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")


def get_paragraph_texts(document_json):
    full_text = document_json.get("text", "")
    paragraphs = []
    for page in document_json.get("pages", []):
        for para in page.get("paragraphs", []):
            segments = para["layout"]["textAnchor"].get("textSegments", [])
            para_text = ""
            for seg in segments:
                start = int(seg.get("startIndex", 0))
                end = int(seg["endIndex"])
                para_text += full_text[start:end]
            para_text = para_text.strip()
            if para_text:
                paragraphs.append(para_text)
    return paragraphs


def estimate_tokens(text):
    # Rough heuristic - no exact tokenizer available here. A real run
    # (ESC guidelines doc) hit the 20,000-token API limit at a batch
    # estimated under 15,000 using ~4 chars/token, meaning this text
    # tokenizes less efficiently than typical English prose (dense
    # abbreviations, diacritics, reference lists). Using ~2 chars/token
    # instead, with a much lower batch cap below, to stay safe
    # regardless of how a given document's text actually tokenizes.
    return len(text) // 2 + 1


def embed_texts(texts, max_batch_size=250, max_batch_tokens=8000):
    # The API caps both: 250 instances per batch AND ~20,000 tokens per
    # batch. Chunk sizes vary a lot, so a fixed instance-count batch can
    # still blow the token cap - flush whenever either limit would be
    # exceeded.
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


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_chunk(paragraphs):
    if not paragraphs:
        return []
    if len(paragraphs) == 1:
        return paragraphs

    embeddings = embed_texts(paragraphs)

    similarities = [
        cosine_similarity(embeddings[i - 1], embeddings[i])
        for i in range(1, len(embeddings))
    ]
    threshold = float(np.percentile(similarities, BREAKPOINT_PERCENTILE))

    chunks = []
    current_chunk_paragraphs = [paragraphs[0]]
    current_chunk_words = len(paragraphs[0].split())

    for i in range(1, len(paragraphs)):
        sim = similarities[i - 1]
        para_words = len(paragraphs[i].split())

        same_topic = sim >= threshold
        fits = (current_chunk_words + para_words) <= MAX_CHUNK_WORDS

        if same_topic and fits:
            current_chunk_paragraphs.append(paragraphs[i])
            current_chunk_words += para_words
        else:
            chunks.append(" ".join(current_chunk_paragraphs))
            current_chunk_paragraphs = [paragraphs[i]]
            current_chunk_words = para_words

    chunks.append(" ".join(current_chunk_paragraphs))
    return chunks


MIN_PARAGRAPH_WORDS = 8


def merge_short_paragraphs(paragraphs, min_words=MIN_PARAGRAPH_WORDS):
    """Document AI splits section labels ('RESULTS'), author lines, and
    footnote fragments into their own paragraphs. A standalone 1-word
    paragraph has no retrievable meaning and breaks similarity continuity
    between the real content around it, so fold any short paragraph into
    the one that follows it (or, if it's the last one, into the one
    before it)."""
    merged = []
    buffer = ""
    for para in paragraphs:
        if buffer:
            para = buffer + " " + para
            buffer = ""
        if len(para.split()) < min_words:
            buffer = para
        else:
            merged.append(para)

    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    return merged


SHARD_PATTERN = re.compile(r"^(.*)-(\d+)\.json$")


def group_shards_by_source(json_blobs):
    """Document AI splits a long PDF's output across multiple shard files
    (e.g. Guideline...-0.json, Guideline...-1.json, ...). Group all shards
    for the same source document together, in page order, so we chunk
    one continuous document instead of several disconnected fragments."""
    groups = defaultdict(list)
    for blob in json_blobs:
        filename = blob.name.split("/")[-1]
        match = SHARD_PATTERN.match(filename)
        if not match:
            continue
        source_name, shard_index = match.group(1), int(match.group(2))
        groups[source_name].append((shard_index, blob))

    for source_name in groups:
        groups[source_name].sort(key=lambda pair: pair[0])

    return groups


def process_all_documents():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)

    blobs = list(bucket.list_blobs(prefix=INPUT_PREFIX))
    json_blobs = [b for b in blobs if b.name.endswith(".json")]

    groups = group_shards_by_source(json_blobs)
    print(f"Found {len(json_blobs)} shard files across {len(groups)} source documents.")

    if ONLY_SOURCE:
        groups = {ONLY_SOURCE: groups[ONLY_SOURCE]}
        print(f"ONLY_SOURCE is set - processing just '{ONLY_SOURCE}' for this test run.")

    for source_name, shard_blobs in groups.items():
        print(f"Processing {source_name} ({len(shard_blobs)} shard(s)) ...")

        paragraphs = []
        for _, blob in shard_blobs:
            document_json = json.loads(blob.download_as_text())
            paragraphs.extend(get_paragraph_texts(document_json))

        paragraphs = merge_short_paragraphs(paragraphs)
        chunks = semantic_chunk(paragraphs)

        output = {
            "source_file": source_name,
            "num_chunks": len(chunks),
            "chunks": [
                {"chunk_id": idx, "text": text} for idx, text in enumerate(chunks)
            ],
        }

        output_blob = bucket.blob(f"{OUTPUT_PREFIX}{source_name}.json")
        output_blob.upload_from_string(
            json.dumps(output, indent=2, ensure_ascii=False),
            content_type="application/json",
        )

        print(
            f"  -> {len(chunks)} chunks written to "
            f"gs://{BUCKET}/{OUTPUT_PREFIX}{source_name}.json"
        )


if __name__ == "__main__":
    process_all_documents()
