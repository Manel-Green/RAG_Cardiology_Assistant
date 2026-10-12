import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

const BACKEND_URL = "https://rag-heart-studies-backend-90354696252.europe-west3.run.app/ask";
const SOURCES_DELIMITER = "⁣⁣SOURCES⁣⁣"; // must match src/app.py exactly

function dedupeSources(sources) {
  const seen = new Set();
  const result = [];
  for (const source of sources || []) {
    if (seen.has(source.source_file)) continue;
    seen.add(source.source_file);
    result.push(source);
  }
  return result;
}

// How fast the typewriter reveal runs - independent of how fast the
// network actually delivers chunks, so short answers / fast connections
// don't make the "streaming" effect invisible.
const REVEAL_CHARS_PER_TICK = 3;
const REVEAL_TICK_MS = 10;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const typewriterRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    return () => clearInterval(typewriterRef.current?.intervalId);
  }, []);

  function updateLastAssistant(update) {
    setMessages((prev) => {
      const next = [...prev];
      next[next.length - 1] = { ...next[next.length - 1], ...update };
      return next;
    });
  }

  // Reveals typewriter.target gradually into the last message, regardless
  // of when/how much of it arrived from the network. Call again whenever
  // more target text is available; it's a no-op if already running.
  function ensureTypewriterRunning() {
    const typewriter = typewriterRef.current;
    if (typewriter.intervalId) return;

    typewriter.intervalId = setInterval(() => {
      const next = Math.min(typewriter.target.length, typewriter.revealed + REVEAL_CHARS_PER_TICK);
      if (next > typewriter.revealed) {
        typewriter.revealed = next;
        updateLastAssistant({ text: typewriter.target.slice(0, next) });
      }

      if (typewriter.revealed >= typewriter.target.length && typewriter.streamDone) {
        clearInterval(typewriter.intervalId);
        typewriter.intervalId = null;
        updateLastAssistant({ text: typewriter.target, sources: typewriter.sources });
      }
    }, REVEAL_TICK_MS);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setInput("");
    setLoading(true);

    const typewriter = { target: "", revealed: 0, streamDone: false, sources: [], intervalId: null };
    typewriterRef.current = typewriter;
    let started = false;

    try {
      const response = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!response.ok) throw new Error(`Request failed (${response.status})`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let sawDelimiter = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        if (sawDelimiter) continue; // rest of buffer is the trailing sources JSON

        const idx = buffer.indexOf(SOURCES_DELIMITER);
        if (idx === -1) {
          typewriter.target = buffer;
        } else {
          typewriter.target = buffer.slice(0, idx);
          sawDelimiter = true;
          buffer = buffer.slice(idx + SOURCES_DELIMITER.length);
        }

        if (!started) {
          started = true;
          setLoading(false);
          setMessages((prev) => [...prev, { role: "assistant", text: "", sources: [] }]);
        }
        ensureTypewriterRunning();
      }

      if (sawDelimiter) {
        try {
          typewriter.sources = dedupeSources(JSON.parse(buffer));
        } catch {
          typewriter.sources = [];
        }
      }
      typewriter.streamDone = true;
      ensureTypewriterRunning();
    } catch (err) {
      clearInterval(typewriter.intervalId);
      const message = `Could not reach the assistant: ${err.message}`;
      if (started) {
        updateLastAssistant({ text: message, sources: [] });
      } else {
        setMessages((prev) => [...prev, { role: "assistant", text: message }]);
      }
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <svg className="ecg-trace" viewBox="0 0 1000 60" preserveAspectRatio="none">
          <path
            id="ecg-path"
            d="M0,30 L300,30 L320,30 L335,8 L350,52 L365,30 L380,30 L1000,30"
            fill="none"
            stroke="#1c4a8a"
            strokeWidth="2"
          />
          <circle r="5" fill="#ffffff">
            <animateMotion dur="3.5s" repeatCount="indefinite">
              <mpath href="#ecg-path" />
            </animateMotion>
          </circle>
        </svg>
        <h1>RAG Agent for Cardiology Research</h1>
        <p>The best assistant for cardiologists</p>
      </header>

      <main className="conversation">
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div className="row row-user" key={i}>
              <div className="bubble">{m.text}</div>
            </div>
          ) : (
            <div className="row row-assistant" key={i}>
              <div className="answer">
                <ReactMarkdown>{m.text}</ReactMarkdown>
              </div>
              {m.sources && m.sources.length > 0 && (
                <div className="sources">
                  {m.sources.map((s, j) => (
                    <div className="source-card" key={j}>
                      <div className="source-eyebrow">📖 EVIDENCE</div>
                      <div className="source-title">{s.title}</div>
                      {s.journal && (
                        <div className="source-journal">{s.journal}</div>
                      )}
                      <div className="source-text">
                        {s.text.slice(0, 280)}...
                      </div>
                      {s.paper_url && (
                        <a
                          className="source-link"
                          href={s.paper_url}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          Read the paper →
                        </a>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
        )}
        {loading && (
          <div className="row row-assistant">
            <div className="answer thinking">Thinking...</div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <form className="composer" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me anything"
        />
        <button type="submit" disabled={loading}>
          ➤
        </button>
      </form>
    </div>
  );
}
