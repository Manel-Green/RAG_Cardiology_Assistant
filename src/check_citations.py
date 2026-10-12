"""
Diagnostic: print the first few chunks of each source paper, so we can
read off the real title/author/journal citation directly from the
indexed data - rather than guessing or relying on memory - before
hardcoding a display-title mapping into the frontend.

Run: python src/check_citations.py
"""

import json

from google.cloud import storage

PROJECT_ID = "manel-portfolio-demos"
BUCKET = "manel-portfolio-demos-rag-heart-studies"
CHUNKS_PREFIX = "processed/chunks/"


def main():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)

    blobs = sorted(
        bucket.list_blobs(prefix=CHUNKS_PREFIX), key=lambda b: b.name
    )

    for blob in blobs:
        if not blob.name.endswith(".json"):
            continue
        data = json.loads(blob.download_as_text())
        source = data["source_file"]
        print("=" * 90)
        print(f"SOURCE: {source}")
        print("-" * 90)
        for chunk in data["chunks"][:3]:
            print(f"[{chunk['chunk_id']}] {chunk['text'][:300]}")
            print()


if __name__ == "__main__":
    main()
