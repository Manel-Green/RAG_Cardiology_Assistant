"""
Phase 5: Local on-demand Streamlit demo interface.

A chat-style page that calls the live Cloud Run backend (Phase 4) and
displays the generated answer plus its citations. Runs locally, on
demand, when you want to demo it - not deployed continuously.

Run: streamlit run src/streamlit_app.py
"""

import time

import requests
import streamlit as st

BACKEND_URL = "https://rag-heart-studies-backend-90354696252.europe-west3.run.app/ask"

STRONG_MATCH_THRESHOLD = 0.25

st.set_page_config(
    page_title="Cardiology RAG Assistant",
    page_icon="\U0001FAC0",
    layout="centered",
)

st.markdown(
    """
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .app-byline {
            font-size: 13px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #8097989;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .app-title {
            font-size: 34px;
            font-weight: 700;
            margin-bottom: 2px;
        }
        .app-tagline {
            font-size: 16px;
            color: #51696A;
            margin-bottom: 28px;
        }
        .source-card {
            border: 1px solid #D7E0E0;
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 10px;
            background: #FCFDFD;
        }
        .source-title {
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 6px;
        }
        .match-strong { color: #1C6B3E; font-weight: 600; font-size: 12.5px; }
        .match-moderate { color: #83641E; font-weight: 600; font-size: 12.5px; }
        .source-score { color: #8097989; font-size: 11px; }
        .source-text { font-size: 13.5px; color: #333; margin-top: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-byline">Manel Rahmouni</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">RAG Agent for Cardiology Research</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-tagline">The best assistant for cardiologists. Ask me anything '
    "from the indexed research studies.</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def match_label(distance):
    if distance <= STRONG_MATCH_THRESHOLD:
        return "Strong match", "match-strong"
    return "Moderate match", "match-moderate"


def render_sources(sources):
    seen = set()
    for source in sources:
        if source["source_file"] in seen:
            continue
        seen.add(source["source_file"])

        label, css_class = match_label(source["distance"])
        preview = source["text"][:300].replace("\n", " ")

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">{source['source_file'].replace('_', ' ')}</div>
                <span class="{css_class}">{label}</span>
                <span class="source-score"> (distance {source['distance']:.3f})</span>
                <div class="source-text">{preview}...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Replay the conversation so far
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

# New question - st.chat_input clears itself automatically after each submit
question = st.chat_input("Ask a question about the indexed cardiology research...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status:
            status.update(label="Retrieving relevant studies...")
            time.sleep(0.3)
            status.update(label="Analyzing evidence...")

            try:
                response = requests.post(BACKEND_URL, json={"question": question}, timeout=60)
                response.raise_for_status()
                result = response.json()
            except Exception as e:
                status.update(label="Something went wrong.", state="error")
                st.error(f"Could not reach the assistant: {e}")
                st.stop()

            status.update(label="Drafting answer...")
            time.sleep(0.2)
            status.update(label="Done.", state="complete")

        st.write(result["answer"])
        sources = result.get("sources", [])
        if sources:
            render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        }
    )
