"""
Step 6c: LLM-as-judge groundedness evaluation - for each test question,
generate a real answer through the actual pipeline, then ask Gemini (a
separate call, acting as judge) whether every claim in that answer is
actually supported by the retrieved excerpts. A different question
from evaluate_retrieval.py's precision/hit@K - that checks whether
retrieval found the right document; this checks whether the generated
answer stays faithful to what it was given.

Note: the judge uses Gemini 2.5 Pro, deliberately different from
GENERATION_MODEL (Flash) - verified GA directly via Vertex AI Model
Garden, so the judge isn't scoring the generator's own blind spots.

Run: python src/evaluate_groundedness.py
"""

import vertexai
from google import genai
from google.cloud import aiplatform
from google.genai import types
from vertexai.language_models import TextEmbeddingModel

from generate_answer import (
    GENERATION_MODEL,
    INDEX_ENDPOINT_RESOURCE_NAME,
    INDEX_LOCATION,
    PROJECT_ID,
    VERTEX_LOCATION,
    answer_question,
    load_chunk_lookup,
)

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether an AI-generated answer is fully \
grounded in the provided source excerpts - meaning every factual claim in the \
answer can be directly verified against the excerpts, with no invented facts, \
statistics, or claims not present in the excerpts.

EXCERPTS:
{excerpts}

QUESTION: {question}

ANSWER TO EVALUATE: {answer}

Respond in exactly this format, nothing else:
VERDICT: GROUNDED or NOT GROUNDED
REASONING: <one or two sentences - if NOT GROUNDED, name exactly which claim is unsupported>"""
# Gemini 3.1 Pro Preview - still in preview status, but confirmed
# actually callable (unlike gemini-2.5-pro, which 404'd despite
# appearing available in Model Garden). Deliberately different from
# GENERATION_MODEL (Flash) so the judge isn't scoring the generator's
# own blind spots.
JUDGE_MODEL = "gemini-3.1-pro-preview"

TEST_QUESTIONS = [
    "Does testing for myocardial viability predict survival benefit from coronary bypass surgery?",
    "How long should ECG monitoring continue after an ischemic stroke to detect atrial fibrillation?",
    "How should acute chest pain be evaluated in the emergency department?",
    "What do current guidelines recommend for diagnosing chest pain?",
    "What is the relationship between type 2 diabetes and cardiovascular mortality?",
    "Does sodium bicarbonate prevent contrast-induced acute kidney injury during angiography?",
]


def judge_groundedness(question, answer_text, retrieved_chunks, genai_client):
    excerpts = "\n\n".join(
        f"[Excerpt {i} - Source: {c['source_file']}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks, start=1)
    )
    prompt = JUDGE_PROMPT_TEMPLATE.format(excerpts=excerpts, question=question, answer=answer_text)

    response = genai_client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    text = response.text.strip()

    verdict = "GROUNDED" if "VERDICT: GROUNDED" in text else "NOT GROUNDED"
    reasoning = text.split("REASONING:", 1)[1].strip() if "REASONING:" in text else text
    return verdict, reasoning


def main():
    vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")

    genai_client = genai.Client(enterprise=True, project=PROJECT_ID, location="global")

    aiplatform.init(project=PROJECT_ID, location=INDEX_LOCATION)
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(INDEX_ENDPOINT_RESOURCE_NAME)

    chunk_lookup = load_chunk_lookup()

    verdicts = []

    for question in TEST_QUESTIONS:
        answer_text, retrieved_chunks = answer_question(
            question, embedding_model, index_endpoint, chunk_lookup, genai_client
        )

        verdict, reasoning = judge_groundedness(question, answer_text, retrieved_chunks, genai_client)
        verdicts.append(verdict)

        print("=" * 90)
        print(f"QUESTION: {question}")
        print(f"ANSWER: {answer_text}")
        print(f"VERDICT: {verdict}")
        print(f"REASONING: {reasoning}")
        print()

    print("=" * 90)
    print("SUMMARY")
    grounded_count = verdicts.count("GROUNDED")
    print(f"  Grounded: {grounded_count}/{len(verdicts)} ({100*grounded_count/len(verdicts):.0f}%)")


if __name__ == "__main__":
    main()