import { useEffect, useRef, useState } from "react";
import ecgBackground from "./assets/ecg-background.jpg";

const BACKEND_URL =
  "https://rag-heart-studies-backend-90354696252.europe-west3.run.app/ask";

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

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSubmit(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer,
          sources: dedupeSources(data.sources),
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Could not reach the assistant: ${err.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className="page"
      style={{ "--bg-photo": `url(${ecgBackground})` }}
    >
      <header className="header">
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
              <div className="answer">{m.text}</div>
              {m.sources && m.sources.length > 0 && (
                <div className="sources">
                  {m.sources.map((s, j) => (
                    <div className="source-card" key={j}>
                      <div className="source-title">
                        {s.source_file.replaceAll("_", " ")}
                      </div>
                      <div className="source-score">
                        distance {s.distance.toFixed(3)}
                      </div>
                      <div className="source-text">
                        {s.text.slice(0, 280)}...
                      </div>
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
