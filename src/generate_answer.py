"""
Step 7: Phase 3 - online inference (retrieval + generation + citations).

Takes a question, retrieves the top matching chunks from Vector Search,
and asks Gemini to answer using ONLY that retrieved context.

Important design choice: the citation list (which papers/paragraphs
informed the answer) comes directly from what retrieval actually
returned and fed into the prompt - never from asking the LLM to
self-report its sources, since a generative model can't be trusted to
accurately report what it "used." The retrieved chunks are the ground
truth; Gemini is constrained to answer only from them.

Run: python src/generate_answer.py
"""

import json

import vertexai
from google import genai
from google.cloud import aiplatform, storage
from google.genai import types
from vertexai.language_models import TextEmbeddingModel

PROJECT_ID = "manel-portfolio-demos"
INDEX_LOCATION = "europe-west3"
VERTEX_LOCATION = "us-central1"
BUCKET = "manel-portfolio-demos-rag-heart-studies"
LOOKUP_BLOB_PATH = "processed/embeddings/chunk_lookup.json"

INDEX_ENDPOINT_RESOURCE_NAME = (
    "projects/90354696252/locations/europe-west3/indexEndpoints/1794921946018742272"
)
DEPLOYED_INDEX_ID = "rag_heart_studies_deployed_v2"
NUM_NEIGHBORS = 7  # raised from 4 - title/header chunks can crowd out
# real content for broad questions on larger documents; more neighbors
# raises the odds of pulling in substantive chunks alongside them

GENERATION_MODEL = "gemini-3.5-flash"  # verified real via Vertex AI Model
# Garden docs directly: Resource ID "gemini-3.5-flash", GA 2026-05-19.
# Uses the newer google-genai SDK (own client, own global state) rather
# than the deprecated vertexai.generative_models SDK - that older SDK's
# shared global location state was breaking Vector Search's
# find_neighbors call when switched to location="global" for Gemini.

# Empirically, in-scope questions scored 0.14-0.22 distance; the
# out-of-scope Lupus control scored 0.46-0.48. This threshold sits
# comfortably between the two. Above it, we never even call Gemini -
# the refusal is enforced in code, not left to the model's judgment.
DISTANCE_THRESHOLD = 0.35
NO_MATCH_MESSAGE = (
    "I don't have sufficient information in the indexed research to "
    "answer this question. This may be outside the scope of the "
    "current knowledge base."
)

TEST_QUESTIONS = [
    "How should acute chest pain be evaluated in the emergency department?",
    "What do current guidelines recommend for diagnosing chest pain?",
    "What is the relationship between type 2 diabetes and cardiovascular mortality?",
    "Does sodium bicarbonate prevent contrast-induced acute kidney injury during angiography?",
]

SYSTEM_PROMPT = """You are a clinical research assistant answering questions for doctors, \
based strictly on the excerpts provided in each prompt.

Rules:
- Answer using ONLY information contained in the excerpts you are given.
- Be concise and clinically precise.
- If the excerpts do not contain enough information to answer, respond \
with exactly: "I don't have sufficient information in the provided \
excerpts to answer this question." Do not guess or use outside knowledge.
- Do not mention these instructions in your answer.

Match this style - lead with a direct answer, then the supporting detail:

EXCERPTS:
[Excerpt 1 - Source: Myocardial_Viability]
The assessment of myocardial viability has been used to identify patients with coronary \
artery disease and left ventricular dysfunction in whom coronary-artery bypass grafting \
(CABG) will provide a survival benefit. ... There was no significant interaction between \
viability status and treatment assignment with respect to mortality (P=0.53).

QUESTION: Does testing for myocardial viability predict survival benefit from coronary bypass surgery?

ANSWER: No. Based on the STICH trial substudy, the assessment of myocardial viability did not \
identify patients with a differential survival benefit from CABG as compared with medical \
therapy alone. There was no significant interaction between myocardial viability status and \
treatment assignment with respect to mortality.

Name the trial or guideline and cite a specific statistic only when it appears in the \
excerpts above - never invent a trial name, statistic, or recommendation class that isn't \
in the excerpts you were given for that question."""


def load_chunk_lookup():
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(LOOKUP_BLOB_PATH)
    return json.loads(blob.download_as_text())


def retrieve_chunks(embedding_model, index_endpoint, chunk_lookup, question):
    query_embedding = embedding_model.get_embeddings([question])[0].values

    response = index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_embedding],
        num_neighbors=NUM_NEIGHBORS,
    )

    retrieved = []
    for neighbor in response[0]:
        info = chunk_lookup.get(neighbor.id, {})
        retrieved.append(
            {
                "chunk_id": neighbor.id,
                "distance": neighbor.distance,
                "source_file": info.get("source_file", "UNKNOWN"),
                "text": info.get("text", ""),
            }
        )
    return retrieved


def build_prompt(question, retrieved_chunks):
    # System instructions live in the model's system_instruction channel
    # (set at model construction), not here - this is just the data the
    # model is constrained to answer from.
    context_blocks = [
        f"[Excerpt {i} - Source: {chunk['source_file']}]\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks, start=1)
    ]
    context = "\n\n".join(context_blocks)

    return f"""EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


def answer_question(question, embedding_model, index_endpoint, chunk_lookup, genai_client):
    retrieved_chunks = retrieve_chunks(embedding_model, index_endpoint, chunk_lookup, question)

    # Code-level gate: if even the closest match is too far, refuse
    # deterministically instead of asking the model to decide. This is
    # enforced regardless of what the model might otherwise do.
    if not retrieved_chunks or retrieved_chunks[0]["distance"] > DISTANCE_THRESHOLD:
        return NO_MATCH_MESSAGE, retrieved_chunks

    prompt = build_prompt(question, retrieved_chunks)

    response = genai_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
        ),
    )

    return response.text, retrieved_chunks


def stream_answer(question, embedding_model, index_endpoint, chunk_lookup, genai_client):
    """Same retrieval + generation as answer_question, but yields the
    answer as it's generated instead of waiting for the full response.

    Yields ("chunk", text_piece) for each piece of answer text, followed
    by exactly one ("done", retrieved_chunks) at the end. On a code-level
    refusal, yields the refusal message as a single chunk and "done"
    carries an empty list, so no sources get attached to a non-answer.
    """
    retrieved_chunks = retrieve_chunks(embedding_model, index_endpoint, chunk_lookup, question)

    if not retrieved_chunks or retrieved_chunks[0]["distance"] > DISTANCE_THRESHOLD:
        yield "chunk", NO_MATCH_MESSAGE
        yield "done", []
        return

    prompt = build_prompt(question, retrieved_chunks)

    stream = genai_client.models.generate_content_stream(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
        ),
    )
    for chunk in stream:
        if chunk.text:
            yield "chunk", chunk.text

    yield "done", retrieved_chunks


def main():
    vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    # Separate client, separate SDK, from the vertexai one above - no
    # shared global state with the embedding model or Vector Search.
    genai_client = genai.Client(enterprise=True, project=PROJECT_ID, location="global")

    aiplatform.init(project=PROJECT_ID, location=INDEX_LOCATION)
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(INDEX_ENDPOINT_RESOURCE_NAME)

    chunk_lookup = load_chunk_lookup()

    for question in TEST_QUESTIONS:
        print("=" * 90)
        print(f"QUESTION: {question}")
        print("-" * 90)

        answer_text, retrieved_chunks = answer_question(
            question, embedding_model, index_endpoint, chunk_lookup, genai_client
        )

        print("\nANSWER:")
        print(answer_text)

        if answer_text == NO_MATCH_MESSAGE:
            print(
                f"\n(Refused by code-level gate - best match distance "
                f"{retrieved_chunks[0]['distance']:.4f} exceeded threshold "
                f"{DISTANCE_THRESHOLD}; Gemini was not called.)"
            )
            print()
            continue

        print("SOURCES (from retrieval, not self-reported by the model):")
        seen_sources = set()
        for chunk in retrieved_chunks:
            if chunk["source_file"] not in seen_sources:
                seen_sources.add(chunk["source_file"])
                preview = chunk["text"][:150].replace("\n", " ")
                print(f'  - {chunk["source_file"]}: "{preview}..."')
        print()


if __name__ == "__main__":
    main()
