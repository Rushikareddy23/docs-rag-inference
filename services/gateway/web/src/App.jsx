import { useState } from "react";

const EXAMPLES = [
  "How does Triton's dynamic batcher combine requests?",
  "What does gpu-memory-utilization control in vLLM?",
  "How do I run vLLM across multiple GPUs?",
];

async function ask(question, mode) {
  const res = await fetch("/v1/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ? JSON.stringify(body.detail) : `HTTP ${res.status}`);
  }
  return res.json();
}

// Turn "[S1]" markers in the answer into links that jump to the matching source card.
function Answer({ text }) {
  const parts = text.split(/(\[S\d+\])/g);
  return (
    <p className="answer">
      {parts.map((p, i) =>
        /^\[S\d+\]$/.test(p) ? (
          <a key={i} className="cite" href={`#src-${p.slice(1, -1)}`}>
            {p}
          </a>
        ) : (
          <span key={i}>{p}</span>
        )
      )}
    </p>
  );
}

function Steps({ steps }) {
  return (
    <ol className="steps">
      {steps.map((s, i) => (
        <li key={i} className={`step step-${s.action}`}>
          {s.action === "search" && (
            <>
              <b>Search</b> “{s.query}” → {s.results} new result{s.results === 1 ? "" : "s"}
            </>
          )}
          {s.action === "verify" && (
            <>
              <b>Evidence check</b>{" "}
              {s.supported === true
                ? "✓ supported"
                : s.supported === false
                ? `✗ missing: ${s.missing || "unspecified"}`
                : "? could not be read (answer is unverified)"}
            </>
          )}
          {s.action === "answer" && (
            <>
              <b>Answer</b>
              {s.forced ? " (step limit reached)" : ""}
              {s.verified === false ? " (not fully supported by the sources)" : ""}
              {s.verified === null ? " (unverified)" : ""}
            </>
          )}
        </li>
      ))}
    </ol>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("agent");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(q = question) {
    if (q.trim().length < 3) return;
    setQuestion(q);
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await ask(q.trim(), mode));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header>
        <h1>Docs Assistant</h1>
        <p className="sub">Answers about vLLM and NVIDIA Triton, grounded in their documentation.</p>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about LLM serving…"
          rows={3}
        />
        <div className="row">
          <div className="modes" role="radiogroup" aria-label="Answer mode">
            {[
              ["agent", "Agent (search → check → answer)"],
              ["rag", "Simple RAG (one pass)"],
            ].map(([value, label]) => (
              <label key={value} className={mode === value ? "on" : ""}>
                <input type="radio" name="mode" value={value} checked={mode === value} onChange={() => setMode(value)} />
                {label}
              </label>
            ))}
          </div>
          <button type="submit" disabled={loading || question.trim().length < 3}>
            {loading ? "Thinking…" : "Ask"}
          </button>
        </div>
      </form>

      {!result && !loading && !error && (
        <div className="examples">
          {EXAMPLES.map((q) => (
            <button key={q} className="chip" onClick={() => submit(q)}>
              {q}
            </button>
          ))}
        </div>
      )}

      {error && <p className="error">Request failed: {error}</p>}

      {result && (
        <section className="result">
          <Answer text={result.answer} />
          <div className="metrics">
            <span>{result.timings_ms.total} ms total</span>
            <span>{result.timings_ms.llm} ms in LLM</span>
            <span>{result.timings_ms.retrieval} ms retrieval</span>
            <span>{result.usage.llm_calls} LLM calls</span>
            <span>{result.usage.completion_tokens} tokens out</span>
          </div>

          <h2>How it got there</h2>
          <Steps steps={result.steps} />

          <h2>Sources</h2>
          <ul className="sources">
            {result.sources.map((s) => (
              <li key={s.ref} id={`src-${s.ref}`}>
                <span className="ref">{s.ref}</span>
                <span className="path">{s.source}</span>
                {s.section && <span className="section">{s.section}</span>}
                <span className="score">{s.score.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
