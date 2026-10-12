"""
Step 6: Query the deployed Vector Search index with real test questions
and confirm it retrieves genuinely relevant chunks.

For each question: embed it with the same model used for the chunks,
ask the endpoint for the top-5 nearest neighbors (chunk IDs + distance
scores only - Vector Search doesn't store the actual text), then look
each ID up in chunk_lookup.json to show the real passage retrieved.

Run: python src/query_index.py
"""

import json

import vertexai
from google.cloud import aiplatform, storage
from vertexai.language_models import TextEmbeddingModel

PROJECT_ID = "manel-portfolio-demos"
LOCATION = "europe-west3"
VERTEX_EMBED_LOCATION = "us-central1"
BUCKET = "manel-portfolio-demos-rag-heart-studies"
LOOKUP_BLOB_PATH = "processed/embeddings/chunk_lookup.json"

INDEX_ENDPOINT_RESOURCE_NAME = (
    "projects/90354696252/locations/europe-west3/indexEndpoints/1794921946018742272"
)
DEPLOYED_INDEX_ID = "rag_heart_studies_deployed_v2"
NUM_NEIGHBORS = 5

TEST_QUESTIONS = [
    "Does testing for myocardial viability predict survival benefit from coronary bypass surgery?",
    "How long should ECG monitoring continue after an ischemic stroke to detect atrial fibrillation?",
    "How should acute chest pain be evaluated in the emergency department?",
    "What do current guidelines recommend for diagnosing chest pain?",
    "What is the relationship between type 2 diabetes and cardiovascular mortality?",
    "Does sodium bicarbonate prevent contrast-induced acute kidney injury during angiography?",
    "What are the latest treatment guidelines for Lupus?",  # out-of-scope control question
]


def load_chunk_lookup():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(LOOKUP_BLOB_PATH)
    return json.loads(blob.download_as_text())


def embed_question(model, question):
    embeddings = model.get_embeddings([question])
    return embeddings[0].values


def main():
    vertexai.init(project=PROJECT_ID, location=VERTEX_EMBED_LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(INDEX_ENDPOINT_RESOURCE_NAME)

    chunk_lookup = load_chunk_lookup()

    for question in TEST_QUESTIONS:
        print("=" * 90)
        print(f"QUESTION: {question}")
        print("-" * 90)

        query_embedding = embed_question(embedding_model, question)

        response = index_endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_embedding],
            num_neighbors=NUM_NEIGHBORS,
        )

        neighbors = response[0]
        for rank, neighbor in enumerate(neighbors, start=1):
            chunk_id = neighbor.id
            distance = neighbor.distance
            info = chunk_lookup.get(chunk_id, {})
            source = info.get("source_file", "UNKNOWN")
            text = info.get("text", "")[:220].replace("\n", " ")
            print(f"  [{rank}] distance={distance:.4f}  source={source}")
            print(f"      {text}...")
        print()


if __name__ == "__main__":
    main()
