"""
Phase 2: Incremental re-indexing pipeline.

Detects PDFs in raw-pdfs/ that haven't been processed yet (no matching
file in processed/chunks/), and runs ONLY those through extraction,
chunking, and embedding - never reprocessing the existing corpus. The
new vectors are added to the already-deployed Vector Search index via
an incremental update (no full rebuild, no redeploy, no downtime).

This is the script a Cloud Scheduler job will trigger on a cron
schedule (e.g. biweekly) once wrapped in a Cloud Run Job. For now,
run it directly to test the mechanism end-to-end:

    python src/reindex_new_studies.py
"""

import json
import time

import vertexai
from google.cloud import aiplatform, documentai_v1 as documentai, storage
from vertexai.language_models import TextEmbeddingModel

from chunk_text import (
    embed_texts,
    group_shards_by_source,
    merge_short_paragraphs,
    semantic_chunk,
    get_paragraph_texts,
)

PROJECT_ID = "manel-portfolio-demos"
DOCAI_LOCATION = "eu"
VERTEX_LOCATION = "us-central1"
INDEX_LOCATION = "europe-west3"

BUCKET = "manel-portfolio-demos-rag-heart-studies"
RAW_PDFS_PREFIX = "raw-pdfs/"
CHUNKS_PREFIX = "processed/chunks/"
LOOKUP_BLOB_PATH = "processed/embeddings/chunk_lookup.json"

DOCAI_PROCESSOR_ID = "1aada585f613aad5"

INDEX_RESOURCE_NAME = (
    "projects/90354696252/locations/europe-west3/indexes/2989501747178766336"
)

RUN_ID = time.strftime("%Y%m%d%H%M%S")
DOCAI_OUTPUT_PREFIX = f"processed/docai-raw-incremental/{RUN_ID}/"
EMBEDDINGS_DELTA_PREFIX = f"processed/embeddings/vectors-delta-{RUN_ID}/"


def find_new_pdfs(bucket):
    raw_blobs = list(bucket.list_blobs(prefix=RAW_PDFS_PREFIX))
    raw_names = {
        b.name.split("/")[-1].rsplit(".pdf", 1)[0]
        for b in raw_blobs
        if b.name.endswith(".pdf")
    }

    chunk_blobs = list(bucket.list_blobs(prefix=CHUNKS_PREFIX))
    already_processed = {
        b.name.split("/")[-1].rsplit(".json", 1)[0] for b in chunk_blobs
    }

    new_names = raw_names - already_processed
    return [
        f"gs://{BUCKET}/{RAW_PDFS_PREFIX}{name}.pdf" for name in sorted(new_names)
    ]


def extract_new_pdfs(new_pdf_uris):
    client_options = {"api_endpoint": f"{DOCAI_LOCATION}-documentai.googleapis.com"}
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)
    processor_name = client.processor_path(PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)

    gcs_documents = documentai.GcsDocuments(
        documents=[
            documentai.GcsDocument(gcs_uri=uri, mime_type="application/pdf")
            for uri in new_pdf_uris
        ]
    )
    input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)

    output_uri = f"gs://{BUCKET}/{DOCAI_OUTPUT_PREFIX}"
    gcs_output = documentai.DocumentOutputConfig.GcsOutputConfig(gcs_uri=output_uri)
    output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output)

    request = documentai.BatchProcessRequest(
        name=processor_name,
        input_documents=input_config,
        document_output_config=output_config,
    )

    print(f"Submitting Document AI batch job for {len(new_pdf_uris)} new file(s)...")
    operation = client.batch_process_documents(request)
    print("Waiting for extraction to complete...")
    operation.result(timeout=600)
    print("Extraction done.")


def chunk_new_pdfs(bucket):
    docai_blobs = [
        b
        for b in bucket.list_blobs(prefix=DOCAI_OUTPUT_PREFIX)
        if b.name.endswith(".json")
    ]
    groups = group_shards_by_source(docai_blobs)

    new_chunk_files = {}
    for source_name, shard_blobs in groups.items():
        print(f"Chunking {source_name} ({len(shard_blobs)} shard(s))...")
        paragraphs = []
        for _, blob in shard_blobs:
            document_json = json.loads(blob.download_as_text())
            paragraphs.extend(get_paragraph_texts(document_json))

        paragraphs = merge_short_paragraphs(paragraphs)
        chunks = semantic_chunk(paragraphs)

        output = {
            "source_file": source_name,
            "num_chunks": len(chunks),
            "chunks": [{"chunk_id": idx, "text": text} for idx, text in enumerate(chunks)],
        }

        chunk_blob_path = f"{CHUNKS_PREFIX}{source_name}.json"
        bucket.blob(chunk_blob_path).upload_from_string(
            json.dumps(output, indent=2, ensure_ascii=False),
            content_type="application/json",
        )
        print(f"  -> {len(chunks)} chunks written to gs://{BUCKET}/{chunk_blob_path}")
        new_chunk_files[source_name] = output["chunks"]

    return new_chunk_files


def embed_new_chunks(bucket, new_chunk_files, embedding_model):
    all_ids = []
    all_texts = []
    new_lookup_entries = {}

    for source_name, chunks in new_chunk_files.items():
        for chunk in chunks:
            chunk_id = f"{source_name}__{chunk['chunk_id']}"
            all_ids.append(chunk_id)
            all_texts.append(chunk["text"])
            new_lookup_entries[chunk_id] = {
                "text": chunk["text"],
                "source_file": source_name,
            }

    print(f"Embedding {len(all_texts)} new chunks...")
    # Reuses chunk_text.py's embed_texts - same token/instance-aware
    # batching, one place to fix instead of three.
    embeddings = embed_texts(all_texts)

    embeddings_lines = [
        json.dumps({"id": chunk_id, "embedding": embedding.tolist()})
        for chunk_id, embedding in zip(all_ids, embeddings)
    ]
    delta_blob_path = f"{EMBEDDINGS_DELTA_PREFIX}embeddings.json"
    bucket.blob(delta_blob_path).upload_from_string(
        "\n".join(embeddings_lines), content_type="application/json"
    )
    print(f"-> wrote {len(embeddings_lines)} new vectors to gs://{BUCKET}/{delta_blob_path}")

    lookup_blob = bucket.blob(LOOKUP_BLOB_PATH)
    existing_lookup = json.loads(lookup_blob.download_as_text())
    existing_lookup.update(new_lookup_entries)
    lookup_blob.upload_from_string(
        json.dumps(existing_lookup, indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    print(f"-> merged {len(new_lookup_entries)} new entries into chunk_lookup.json")


def update_live_index():
    aiplatform.init(project=PROJECT_ID, location=INDEX_LOCATION)
    index = aiplatform.MatchingEngineIndex(INDEX_RESOURCE_NAME)

    delta_uri = f"gs://{BUCKET}/{EMBEDDINGS_DELTA_PREFIX}"
    print(f"Updating live index incrementally from {delta_uri} ...")
    index.update_embeddings(contents_delta_uri=delta_uri, is_complete_overwrite=False)
    print("Index update submitted. New vectors should become queryable shortly.")


def main():
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET)

    new_pdf_uris = find_new_pdfs(bucket)
    if not new_pdf_uris:
        print("No new PDFs found - everything in raw-pdfs/ is already indexed.")
        return

    print(f"Found {len(new_pdf_uris)} new PDF(s): {new_pdf_uris}")

    extract_new_pdfs(new_pdf_uris)

    new_chunk_files = chunk_new_pdfs(bucket)

    vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    embed_new_chunks(bucket, new_chunk_files, embedding_model)

    update_live_index()

    print("\nDone. The new document(s) should be queryable once the index update finishes.")


if __name__ == "__main__":
    main()
