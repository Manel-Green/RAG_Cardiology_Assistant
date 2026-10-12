"""
Diagnostic: look at the actual chunk content of Outcomes_after_Angiography
to understand what the paper is really about - its chunks didn't appear
in the top-5 for the angiography question, so let's see if the question
phrasing just doesn't match how the paper is actually written.

Run: python src/peek_angiography.py
"""

import json

from google.cloud import storage

PROJECT_ID = "manel-portfolio-demos"
BUCKET = "manel-portfolio-demos-rag-heart-studies"
BLOB_PATH = "processed/chunks/Outcomes_after_Angiography.json"


def peek():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(BLOB_PATH)
    data = json.loads(blob.download_as_text())

    print("source_file:", data["source_file"])
    print("num_chunks:", data["num_chunks"])
    for c in data["chunks"][:6]:
        print(f"\n--- chunk {c['chunk_id']} ---")
        print(c["text"][:400])


if __name__ == "__main__":
    peek()
