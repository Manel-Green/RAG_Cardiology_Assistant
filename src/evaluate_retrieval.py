"""
Step 6b: Formal retrieval evaluation - Precision@K and Hit@K against a
known-correct-source test set, instead of manually eyeballing printed
results (query_index.py). Reuses the same test questions, now paired
with ground truth.

Run: python src/evaluate_retrieval.py
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
DISTANCE_THRESHOLD = 0.35  # must match generate_answer.py's refusal gate

# Each in-scope question paired with the source_file it should retrieve
# from (ground truth). The control question has no correct source - it
# should be refused instead, so its expected value is None.
TEST_CASES = [
    ("Does testing for myocardial viability predict survival benefit from coronary bypass surgery?",
     "Myocardial_Viability"),
    ("How long should ECG monitoring continue after an ischemic stroke to detect atrial fibrillation?",
     "Cryptogenic_Stroke"),
    ("How should acute chest pain be evaluated in the emergency department?",
     "Acute_Chest_Pain_Emergency_Department"),
    ("What do current guidelines recommend for diagnosing chest pain?",
     "Guideline_Evaluation_and_Diagnosis_Chest_Pain"),
    ("What is the relationship between type 2 diabetes and cardiovascular mortality?",
     "Cardiovascular_Outcomes_Mortality_Type2 _Diabetes"),
    ("Does sodium bicarbonate prevent contrast-induced acute kidney injury during angiography?",
     "Outcomes_after_Angiography"),
    ("What are the latest treatment guidelines for Lupus?", None),  # out-of-scope control
]


def load_chunk_lookup():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(LOOKUP_BLOB_PATH)
    return json.loads(blob.download_as_text())


def retrieve(embedding_model, index_endpoint, chunk_lookup, question):
    query_embedding = embedding_model.get_embeddings([question])[0].values
    response = index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_embedding],
        num_neighbors=NUM_NEIGHBORS,
    )
    results = []
    for neighbor in response[0]:
        info = chunk_lookup.get(neighbor.id, {})
        results.append({
            "distance": neighbor.distance,
            "source_file": info.get("source_file", "UNKNOWN"),
        })
    return results


def evaluate_in_scope(retrieved, expected_source):
    hits = [r for r in retrieved if r["source_file"] == expected_source]
    precision_at_k = len(hits) / len(retrieved) if retrieved else 0.0
    hit_at_k = len(hits) > 0
    return precision_at_k, hit_at_k


def evaluate_control(retrieved):
    if not retrieved:
        return True
    return retrieved[0]["distance"] > DISTANCE_THRESHOLD


def main():
    vertexai.init(project=PROJECT_ID, location=VERTEX_EMBED_LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(INDEX_ENDPOINT_RESOURCE_NAME)

    chunk_lookup = load_chunk_lookup()

    precisions = []
    hits = []
    control_correct = None

    for question, expected_source in TEST_CASES:
        retrieved = retrieve(embedding_model, index_endpoint, chunk_lookup, question)

        print("=" * 90)
        print(f"QUESTION: {question}")
        print(f"EXPECTED SOURCE: {expected_source or '(none - out-of-scope control)'}")
        for rank, r in enumerate(retrieved, start=1):
            print(f"  [{rank}] distance={r['distance']:.4f}  source={r['source_file']}")

        if expected_source is None:
            control_correct = evaluate_control(retrieved)
            print(f"RESULT: {'PASS - correctly refusable' if control_correct else 'FAIL - should have refused'}")
        else:
            precision_at_k, hit_at_k = evaluate_in_scope(retrieved, expected_source)
            precisions.append(precision_at_k)
            hits.append(hit_at_k)
            print(f"RESULT: precision@{NUM_NEIGHBORS}={precision_at_k:.2f}  hit@{NUM_NEIGHBORS}={'yes' if hit_at_k else 'no'}")
        print()

    print("=" * 90)
    print("SUMMARY")
    print(f"  Mean precision@{NUM_NEIGHBORS} across {len(precisions)} in-scope questions: {sum(precisions)/len(precisions):.2f}")
    print(f"  Hit@{NUM_NEIGHBORS} rate: {sum(hits)}/{len(hits)} ({100*sum(hits)/len(hits):.0f}%)")
    print(f"  Control question correctly refusable: {'yes' if control_correct else 'no'}")


if __name__ == "__main__":
    main()