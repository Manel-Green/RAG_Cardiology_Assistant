"""
One-time setup script: creates a Document AI OCR processor.

Document AI processors can't be created via the gcloud CLI (no command
group for it), so this uses the Python client library instead.

Run once: python src/create_processor.py
After running, copy the printed Processor ID — the extraction script
will need it.
"""

from google.cloud import documentai_v1 as documentai

PROJECT_ID = "manel-portfolio-demos"
LOCATION = "eu"  # matches the bucket's europe-west3 region
PROCESSOR_DISPLAY_NAME = "rag-heart-studies-ocr"


def create_processor():
    client_options = {"api_endpoint": f"{LOCATION}-documentai.googleapis.com"}
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)

    parent = client.common_location_path(PROJECT_ID, LOCATION)

    processor = documentai.Processor(
        display_name=PROCESSOR_DISPLAY_NAME,
        type_="OCR_PROCESSOR",
    )

    result = client.create_processor(parent=parent, processor=processor)

    processor_id = result.name.split("/")[-1]
    print(f"Processor created.")
    print(f"Full resource name: {result.name}")
    print(f"Processor ID: {processor_id}")


if __name__ == "__main__":
    create_processor()
