"""
Step 2a: Extract text from every PDF in raw-pdfs/ using the Document AI
OCR processor, via Document AI's batch processing API (not the simple
single-file API, which caps out around 15 pages per PDF).

Reads from:  gs://manel-portfolio-demos-rag-heart-studies/raw-pdfs/
Writes to:   gs://manel-portfolio-demos-rag-heart-studies/processed/docai-raw/

Run: python src/extract_text.py
This submits a batch job and blocks until it finishes (a few minutes
for 6 PDFs).
"""

from google.cloud import documentai_v1 as documentai

PROJECT_ID = "manel-portfolio-demos"
LOCATION = "eu"
PROCESSOR_ID = "1aada585f613aad5"
BUCKET = "manel-portfolio-demos-rag-heart-studies"
INPUT_PREFIX = f"gs://{BUCKET}/raw-pdfs/"
OUTPUT_PREFIX = f"gs://{BUCKET}/processed/docai-raw/"


def extract_text():
    client_options = {"api_endpoint": f"{LOCATION}-documentai.googleapis.com"}
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)

    processor_name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

    gcs_input = documentai.GcsPrefix(gcs_uri_prefix=INPUT_PREFIX)
    input_config = documentai.BatchDocumentsInputConfig(gcs_prefix=gcs_input)

    gcs_output = documentai.DocumentOutputConfig.GcsOutputConfig(gcs_uri=OUTPUT_PREFIX)
    output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output)

    request = documentai.BatchProcessRequest(
        name=processor_name,
        input_documents=input_config,
        document_output_config=output_config,
    )

    print("Submitting batch processing job...")
    operation = client.batch_process_documents(request)

    print(f"Operation started: {operation.operation.name}")
    print("Waiting for it to complete (this can take a few minutes)...")
    operation.result(timeout=600)

    print("Done. Extracted text written to:", OUTPUT_PREFIX)


if __name__ == "__main__":
    extract_text()
