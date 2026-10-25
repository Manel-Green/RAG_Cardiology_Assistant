"""
Undeploys the Vector Search index, stopping the ~$0.094/hour billing
from deploy_index.py. The index itself (and all its data) is untouched
and free to keep - only the serving machine goes away. Redeploying
later takes the same ~20-30 min as the original deploy.

Run: python src/undeploy_index.py
"""

from google.cloud import aiplatform

PROJECT_ID = "manel-portfolio-demos"
LOCATION = "europe-west3"
INDEX_ENDPOINT_RESOURCE_NAME = (
    "projects/90354696252/locations/europe-west3/indexEndpoints/1794921946018742272"
)
DEPLOYED_INDEX_ID = "rag_heart_studies_deployed_v2"

aiplatform.init(project=PROJECT_ID, location=LOCATION)


def undeploy_index():
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(INDEX_ENDPOINT_RESOURCE_NAME)
    print("Undeploying - this can take a few minutes...")
    index_endpoint.undeploy_index(deployed_index_id=DEPLOYED_INDEX_ID)
    print("Undeployed. Billing for the deployed index has stopped.")


if __name__ == "__main__":
    undeploy_index()