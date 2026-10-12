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

import json

import vertexai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.cloud import aiplatform
from pydantic import BaseModel
from vertexai.language_models import TextEmbeddingModel

from generate_answer import (
    INDEX_ENDPOINT_RESOURCE_NAME,
    INDEX_LOCATION,
    PROJECT_ID,
    VERTEX_LOCATION,
    load_chunk_lookup,
    stream_answer,
)

# Separates the streamed answer text from the trailing sources JSON in
# the response body. Uses invisible-separator code points so it can
# never collide with text Gemini would plausibly generate.
SOURCES_DELIMITER = "⁣⁣SOURCES⁣⁣"

app = FastAPI(title="RAG Heart Studies API")

# Allows the React frontend (running on localhost during dev) to call
# this API - fine to leave open since this serves only public research
# paper content, nothing sensitive.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

# Separate client/SDK from the vertexai one above - see generate_answer.py
# for why (old SDK's shared global state broke Vector Search when
# switched to location="global" for the newer Gemini model).
genai_client = genai.Client(enterprise=True, project=PROJECT_ID, location="global")

aiplatform.init(project=PROJECT_ID, location=INDEX_LOCATION)
index_endpoint = aiplatform.MatchingEngineIndexEndpoint(INDEX_ENDPOINT_RESOURCE_NAME)

chunk_lookup = load_chunk_lookup()

# Verified directly from each paper's own title/author/journal chunk
# (processed/chunks/*.json, chunk_id 0-2) - not guessed or recalled
# from memory. URLs are real DOIs, confirmed via web search against
# the exact title/authors/journal/year, not guessed.
CITATIONS = {
    "Acute_Chest_Pain_Emergency_Department": {
        "title": "2022 ACC Expert Consensus Decision Pathway on the Evaluation and "
        "Disposition of Acute Chest Pain in the Emergency Department",
        "journal": "Journal of the American College of Cardiology, Vol. 80, No. 20, 2022",
        "url": "https://doi.org/10.1016/j.jacc.2022.08.750",
    },
    "Cardiovascular_Outcomes_Mortality_Type2 _Diabetes": {
        "title": "Empagliflozin, Cardiovascular Outcomes, and Mortality in Type 2 Diabetes",
        "journal": "New England Journal of Medicine (EMPA-REG OUTCOME trial)",
        "url": "https://doi.org/10.1056/NEJMoa1504720",
    },
    "Cryptogenic_Stroke": {
        "title": "Cryptogenic Stroke and Underlying Atrial Fibrillation",
        "journal": "N Engl J Med 2014;370:2478-86 (CRYSTAL AF trial)",
        "url": "https://doi.org/10.1056/NEJMoa1313600",
    },
    "ESC_guidelines_prevention_sudden_cardiac_death": {
        "title": "2022 ESC Guidelines for the management of patients with ventricular "
        "arrhythmias and the prevention of sudden cardiac death",
        "journal": "European Heart Journal (2022) 43, 3997-4126",
        "url": "https://doi.org/10.1093/eurheartj/ehac262",
    },
    "Guideline_Evaluation_and_Diagnosis_Chest_Pain": {
        "title": "2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR Guideline for the Evaluation "
        "and Diagnosis of Chest Pain",
        "journal": "Journal of the American College of Cardiology, 2021",
        "url": "https://doi.org/10.1016/j.jacc.2021.07.052",
    },
    "Myocardial_Viability": {
        "title": "Myocardial Viability and Survival in Ischemic Left Ventricular Dysfunction",
        "journal": "New England Journal of Medicine (STICH trial)",
        "url": "https://doi.org/10.1056/NEJMoa1100358",
    },
    "Outcomes_after_Angiography": {
        "title": "Outcomes after Angiography with Sodium Bicarbonate and Acetylcysteine",
        "journal": "N Engl J Med. 2018 Feb 15;378(7) (PRESERVE trial)",
        "url": "https://doi.org/10.1056/NEJMoa1710933",
    },
}

STRONG_MATCH_THRESHOLD = 0.25


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def health_check():
    return {"status": "ok"}


def _build_sources(retrieved_chunks):
    sources = []
    for chunk in retrieved_chunks:
        citation = CITATIONS.get(
            chunk["source_file"],
            {"title": chunk["source_file"].replace("_", " "), "journal": "", "url": None},
        )
        sources.append(
            {
                "source_file": chunk["source_file"],
                "title": citation["title"],
                "journal": citation["journal"],
                "text": chunk["text"],
                "match_confidence": (
                    "Strong match"
                    if chunk["distance"] <= STRONG_MATCH_THRESHOLD
                    else "Moderate match"
                ),
                "paper_url": citation.get("url"),
            }
        )
    return sources


@app.post("/ask")
def ask(request: QuestionRequest):
    def event_stream():
        retrieved_chunks = []
        for event_type, payload in stream_answer(
            request.question,
            embedding_model,
            index_endpoint,
            chunk_lookup,
            genai_client,
        ):
            if event_type == "chunk":
                yield payload
            else:  # "done"
                retrieved_chunks = payload

        yield SOURCES_DELIMITER + json.dumps(_build_sources(retrieved_chunks))

    return StreamingResponse(event_stream(), media_type="text/plain")
