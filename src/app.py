"""
Phase 4: FastAPI backend wrapping the retrieval + generation logic
from generate_answer.py behind an HTTP endpoint, so it can be deployed
on Cloud Run instead of run as a local script.

Models and clients are initialized once at module load (when the
container starts), not per-request - re-creating them on every request
would be slow and wasteful.

Local run (for testing before deploying):
    uvicorn src.app:app --reload --port 8080
Then: curl -X POST localhost:8080/ask -H "Content-Type: application/json" \
    -d '{"question": "Does testing for myocardial viability predict survival benefit from coronary bypass surgery?"}'
"""

import vertexai
from fastapi import FastAPI
from google.cloud import aiplatform
from pydantic import BaseModel
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

from generate_answer import (
    GENERATION_MODEL,
    INDEX_ENDPOINT_RESOURCE_NAME,
    INDEX_LOCATION,
    NO_MATCH_MESSAGE,
    PROJECT_ID,
    SYSTEM_PROMPT,
    VERTEX_LOCATION,
    answer_question,
    load_chunk_lookup,
)

app = FastAPI(title="RAG Heart Studies API")

vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
generation_model = GenerativeModel(GENERATION_MODEL, system_instruction=[SYSTEM_PROMPT])

aiplatform.init(project=PROJECT_ID, location=INDEX_LOCATION)
index_endpoint = aiplatform.MatchingEngineIndexEndpoint(INDEX_ENDPOINT_RESOURCE_NAME)

chunk_lookup = load_chunk_lookup()


class QuestionRequest(BaseModel):
    question: str


class SourceCitation(BaseModel):
    source_file: str
    text: str
    distance: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    answer_text, retrieved_chunks = answer_question(
        request.question,
        embedding_model,
        index_endpoint,
        chunk_lookup,
        generation_model,
    )

    # If the refusal gate triggered, these chunks were retrieved but
    # never actually used to generate an answer - don't present them
    # as if they informed a response that doesn't exist.
    if answer_text == NO_MATCH_MESSAGE:
        return AnswerResponse(answer=answer_text, sources=[])

    sources = [
        SourceCitation(
            source_file=chunk["source_file"],
            text=chunk["text"],
            distance=chunk["distance"],
        )
        for chunk in retrieved_chunks
    ]

    return AnswerResponse(answer=answer_text, sources=sources)
