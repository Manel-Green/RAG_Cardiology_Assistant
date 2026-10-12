"""
Step 4: Create the Vertex AI Vector Search index from the embeddings.

Uses Tree-AH (approximate nearest-neighbor) rather than brute-force.
Brute-force creation hit an unresolvable opaque FAILED_PRECONDITION
error (a known rough edge - see googleapis/python-aiplatform#2474).
Tree-AH is the far more common, battle-tested code path in the SDK,
and at under 1,000 vectors its approximation error is negligible - in
practice it returns the same top results brute-force would.

This step only CREATES the index (a one-time ~$0.05 build fee based on
data size). It is not deployed/queryable yet, and no hourly billing
starts until we deploy it to an endpoint in the next step.

Run: python src/create_index.py
This can take several minutes to finish building.
"""

from google.cloud import aiplatform

PROJECT_ID = "manel-portfolio-demos"
LOCATION = "europe-west3"  # must match the bucket's region exactly
BUCKET_URI = "gs://manel-portfolio-demos-rag-heart-studies"
CONTENTS_DELTA_URI = f"{BUCKET_URI}/processed/embeddings/vectors/"
DIMENSIONS = 768  # text-embedding-005's default output dimension

aiplatform.init(project=PROJECT_ID, location=LOCATION)


def create_index():
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name="rag-heart-studies-index-small",
        contents_delta_uri=CONTENTS_DELTA_URI,
        dimensions=DIMENSIONS,
        approximate_neighbors_count=10,
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=10,
        distance_measure_type="COSINE_DISTANCE",
        shard_size="SHARD_SIZE_SMALL",
        description="Tree-AH index (small shard) over heart-studies RAG chunk embeddings",
    )

    print("Index created.")
    print(f"Index resource name: {index.resource_name}")


if __name__ == "__main__":
    create_index()
