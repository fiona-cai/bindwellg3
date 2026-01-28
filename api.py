from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
HEADING_CHUNKS_PATH = os.path.join(BASE_DIR, "heading-chunks.json")
TABLES_PATH = os.path.join(BASE_DIR, "processed_document_tables.json")


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]


@dataclass(frozen=True)
class HeadingChunk:
    section_index: int
    source: str
    content: str


@dataclass(frozen=True)
class TableSummary:
    table_id: str
    page_number: int
    headers: List[str]
    row_count: int
    column_count: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=10)


class AskResult(BaseModel):
    section_index: int
    score: float
    snippet: str
    content: str
    source: str


class AskResponse(BaseModel):
    question: str
    top_k: int
    results: List[AskResult]


def _load_heading_chunks() -> List[HeadingChunk]:
    if not os.path.exists(HEADING_CHUNKS_PATH):
        raise FileNotFoundError(f"Missing file: {HEADING_CHUNKS_PATH}")
    with open(HEADING_CHUNKS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    chunks: List[HeadingChunk] = []
    for item in raw:
        meta = item.get("metadata") or {}
        chunks.append(
            HeadingChunk(
                section_index=int(meta.get("section_index") or 0),
                source=str(meta.get("source") or ""),
                content=str(item.get("content") or ""),
            )
        )
    return chunks


def _load_tables() -> List[Dict[str, Any]]:
    if not os.path.exists(TABLES_PATH):
        raise FileNotFoundError(f"Missing file: {TABLES_PATH}")
    with open(TABLES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return list(raw.get("tables") or [])


@lru_cache(maxsize=1)
def _bm25_index() -> Tuple[List[HeadingChunk], List[Dict[str, int]], Dict[str, int], List[int], float]:
    """
    Returns:
      - chunks
      - term_freqs per doc: [{term: tf}]
      - doc_freqs: {term: df}
      - doc_lens: [len(tokens)]
      - avgdl
    """
    chunks = _load_heading_chunks()
    term_freqs: List[Dict[str, int]] = []
    doc_freqs: Dict[str, int] = {}
    doc_lens: List[int] = []

    for c in chunks:
        tokens = _tokenize(c.content)
        doc_lens.append(len(tokens))
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        term_freqs.append(tf)
        for t in tf.keys():
            doc_freqs[t] = doc_freqs.get(t, 0) + 1

    avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
    return chunks, term_freqs, doc_freqs, doc_lens, avgdl


def _bm25_score(query_tokens: List[str], doc_tf: Dict[str, int], doc_len: int, doc_freqs: Dict[str, int], n_docs: int, avgdl: float) -> float:
    # Standard-ish BM25 parameters.
    k1 = 1.5
    b = 0.75
    score = 0.0
    for t in query_tokens:
        df = doc_freqs.get(t, 0)
        if df == 0:
            continue
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        tf = doc_tf.get(t, 0)
        if tf == 0:
            continue
        denom = tf + k1 * (1.0 - b + b * (doc_len / (avgdl or 1.0)))
        score += idf * (tf * (k1 + 1.0)) / (denom or 1.0)
    return float(score)


def _make_snippet(text: str, query_tokens: List[str], max_len: int = 240) -> str:
    s = re.sub(r"\s+", " ", (text or "")).strip()
    if not s:
        return ""
    if not query_tokens:
        return (s[:max_len] + ("…" if len(s) > max_len else ""))

    lower = s.lower()
    best_pos = None
    for t in query_tokens[:10]:
        pos = lower.find(t)
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_pos = pos
    if best_pos is None:
        return (s[:max_len] + ("…" if len(s) > max_len else ""))

    start = max(0, best_pos - max_len // 3)
    end = min(len(s), start + max_len)
    snippet = s[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(s):
        snippet = snippet + "…"
    return snippet


@lru_cache(maxsize=1)
def _tables_by_id() -> Dict[str, Dict[str, Any]]:
    tables = _load_tables()
    return {str(t.get("table_id")): t for t in tables if t.get("table_id")}


app = FastAPI(title="Bindwell Document API", version="0.1.0")


# Serve the minimal UI.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index_path)


@app.get("/style.css")
def style_css() -> FileResponse:
    css_path = os.path.join(FRONTEND_DIR, "style.css")
    if not os.path.exists(css_path):
        raise HTTPException(status_code=404, detail="frontend/style.css not found")
    return FileResponse(css_path, media_type="text/css; charset=utf-8")


@app.get("/script.js")
def script_js() -> Response:
    """
    Serve a tiny client that talks to the API endpoints, without editing any files
    under frontend/.

    Commands:
      - normal text: asks /api/ask
      - /tables [query]: lists matching tables
      - /table <table_id>: fetches and prints a table
    """
    js = r"""
function _appendMessage(text, role) {
  const chatContainer = document.getElementById("chat-container");
  const el = document.createElement("div");
  el.classList.add("message", role);
  el.innerText = text;
  chatContainer.appendChild(el);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return el;
}

async function _postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }
  return await res.json();
}

async function _getJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }
  return await res.json();
}

function _formatAskResponse(data) {
  if (!data || !Array.isArray(data.results) || data.results.length === 0) {
    return "No matches found. Try rephrasing your question.";
  }
  const top = data.results[0];
  let out = "";
  out += `Top match (section ${top.section_index}, score ${top.score}):\n`;
  out += `${top.snippet || ""}\n\n`;
  out += "Other matches:\n";
  for (let i = 0; i < Math.min(3, data.results.length); i++) {
    const r = data.results[i];
    out += `- section ${r.section_index} (score ${r.score})\n`;
  }
  out += "\nTip: use /tables [query] to browse extracted tables.";
  return out.trim();
}

function _formatTablesList(data) {
  const tables = (data && data.tables) ? data.tables : [];
  if (!tables.length) return "No tables matched.";
  let out = "Tables (showing up to your limit):\n";
  for (const t of tables) {
    const header = (t.headers && t.headers.length) ? t.headers[0] : "(no header)";
    out += `- ${t.table_id} (p.${t.page_number}) rows=${t.row_count} cols=${t.column_count}\n  ${header}\n`;
  }
  out += "\nTip: use /table <table_id> to view a specific table.";
  return out.trim();
}

function _formatTableDetail(t) {
  if (!t) return "Table not found.";
  let out = `${t.table_id} (page ${t.page_number})\n`;
  if (t.headers && t.headers.length) {
    out += `Headers: ${t.headers.join(" | ")}\n`;
  }
  out += "\n";
  const rows = t.rows || [];
  const maxRows = Math.min(15, rows.length);
  for (let i = 0; i < maxRows; i++) {
    out += `- ${rows[i].join(" | ")}\n`;
  }
  if (rows.length > maxRows) out += `… (${rows.length - maxRows} more rows)\n`;
  return out.trim();
}

async function sendMessage() {
  const input = document.getElementById("user-input");
  const message = (input.value || "").trim();
  if (message === "") return;

  _appendMessage(message, "user");
  const agentEl = _appendMessage("Thinking…", "agent");
  input.value = "";

  try {
    // /tables [query]
    if (message.toLowerCase().startsWith("/tables")) {
      const q = message.slice("/tables".length).trim();
      const url = "/api/tables?limit=10" + (q ? `&query=${encodeURIComponent(q)}` : "");
      const data = await _getJson(url);
      agentEl.innerText = _formatTablesList(data);
      return;
    }

    // /table <id>
    if (message.toLowerCase().startsWith("/table")) {
      const id = message.slice("/table".length).trim();
      if (!id) {
        agentEl.innerText = "Usage: /table <table_id>";
        return;
      }
      const data = await _getJson(`/api/tables/${encodeURIComponent(id)}`);
      agentEl.innerText = _formatTableDetail(data);
      return;
    }

    // Default: ask question
    const data = await _postJson("/api/ask", { question: message, top_k: 5 });
    agentEl.innerText = _formatAskResponse(data);
  } catch (err) {
    agentEl.innerText = `Error: ${err && err.message ? err.message : String(err)}`;
  }
}

document.getElementById("user-input").addEventListener("keydown", function (event) {
  if (event.key === "Enter") {
    sendMessage();
  }
});
"""
    return Response(content=js, media_type="text/javascript; charset=utf-8")


@app.get("/health")
def health() -> Dict[str, Any]:
    # Helpful for quick sanity checks.
    chunks, _, _, _, _ = _bm25_index()
    return {
        "ok": True,
        "heading_chunks": len(chunks),
        "tables": len(_tables_by_id()),
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    query_tokens = _tokenize(req.question)
    chunks, term_freqs, doc_freqs, doc_lens, avgdl = _bm25_index()
    n_docs = len(chunks)
    if n_docs == 0:
        raise HTTPException(status_code=500, detail="No heading chunks loaded")

    scored: List[Tuple[float, int]] = []
    for i, (tf, dl) in enumerate(zip(term_freqs, doc_lens)):
        s = _bm25_score(query_tokens, tf, dl, doc_freqs, n_docs, avgdl)
        if s > 0:
            scored.append((s, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: req.top_k]
    results: List[AskResult] = []
    for score, idx in top:
        c = chunks[idx]
        results.append(
            AskResult(
                section_index=c.section_index,
                score=round(score, 6),
                snippet=_make_snippet(c.content, query_tokens),
                content=c.content,
                source=c.source,
            )
        )

    return AskResponse(question=req.question, top_k=req.top_k, results=results)


@app.get("/api/tables")
def list_tables(query: Optional[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    q_tokens = _tokenize(query or "")

    tables = list(_tables_by_id().values())
    summaries: List[Tuple[float, TableSummary]] = []
    for t in tables:
        headers = list(t.get("headers") or [])
        rows = list(t.get("rows") or [])
        row_count = len(rows)
        column_count = len(headers) if headers else (len(rows[0]) if rows else 0)

        haystack = " ".join(headers)
        if rows:
            # Only sample the first few rows to keep search fast.
            sample_rows = rows[: min(10, len(rows))]
            haystack += " " + " ".join(" ".join(map(str, r)) for r in sample_rows)

        if q_tokens:
            hay_tokens = set(_tokenize(haystack))
            score = float(sum(1 for qt in q_tokens if qt in hay_tokens))
            if score <= 0:
                continue
        else:
            score = 1.0

        summaries.append(
            (
                score,
                TableSummary(
                    table_id=str(t.get("table_id")),
                    page_number=int(t.get("page_number") or 0),
                    headers=[str(h) for h in headers],
                    row_count=row_count,
                    column_count=column_count,
                ),
            )
        )

    summaries.sort(key=lambda x: (x[0], x[1].page_number), reverse=True)
    page = summaries[offset : offset + limit]

    return {
        "query": query or "",
        "offset": offset,
        "limit": limit,
        "total": len(summaries),
        "tables": [
            {
                "table_id": s.table_id,
                "page_number": s.page_number,
                "headers": s.headers,
                "row_count": s.row_count,
                "column_count": s.column_count,
            }
            for _, s in page
        ],
    }


@app.get("/api/tables/{table_id}")
def get_table(table_id: str) -> Dict[str, Any]:
    table = _tables_by_id().get(table_id)
    if not table:
        raise HTTPException(status_code=404, detail=f"Table not found: {table_id}")
    return table


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
    )

