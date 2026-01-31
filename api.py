from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from langchain_core.documents import Document  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
import openai  # noqa: E402

from retrieval.retrieval_langchain import retrieve_chunks  # noqa: E402

from config import MODIFY_QUERY, APP_HOST, APP_PORT

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
HEADING_CHUNKS_PATH = os.path.join(BASE_DIR, "data", "heading-chunks.json")
TABLES_PATH = os.path.join(BASE_DIR, "data", "processed_document_tables.json")

SYSTEM_PROMPT = """You are a careful assistant answering questions about the 2026 EPA Pesticide General Permit (PGP) using ONLY the provided excerpts.

Important:"
- Think through the problem step-by-step, but DO NOT reveal your internal reasoning.
- If the excerpts do not contain enough information to answer, say so and ask a concise clarifying question.
- Never invent permit requirements, thresholds, definitions, examples, or “typical” practices not present in the excerpts.
- Do not mention specific pesticide products/ingredients, thresholds, timelines, or exemptions unless the excerpts explicitly state them.
- Do not deviate from the user's question topic even if the document contains additional irrelevant information.

Reason (privately) in these steps:
1) Identify the pesticide-related subject of the user’s question (e.g., pest/use pattern such as mosquito control, weed/algae, forest canopy; activity; permit requirement; reporting; NOI/NOT; monitoring; discharge conditions; listed species/ESA; etc.).
2) Classify the question as general vs. specific (specific = asks about a particular requirement, form, threshold, who must do what, when, or where).
3) Decide what subdetails would be needed to answer (e.g., operator/decision-maker, treatment area, waters of the U.S., use pattern, pesticide product changes, timing, reporting/records, applicability/coverage).
4) Answer using only excerpt text as evidence.
5) Provide citations as bracketed numbers like [1], [2] that correspond to the excerpt numbers provided.

Output format:
- Answer:
  - Use bullet points for requirements/constraints.
  - Every bullet MUST end with at least one citation like [1] or [2].
  - If you cannot find explicit support in the excerpts for a point, do not include it.
  - If the excerpts are insufficient overall, write: "I don’t have enough in the provided excerpts to answer." and ask 1-2 clarifying questions.
- Citations: (list the bracketed citations you used; if none, write "none")
"""


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
    top_k: int = Field(5, ge=1, le=100)


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


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=10)


class ChatCitation(BaseModel):
    ref: int
    section_index: int
    source: str
    heading_title: str

class RetrievedSection(BaseModel):
    section_index: int
    source: str
    heading_title: str
    content: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[ChatCitation]
    retrieved_sections: List[RetrievedSection]
    top_k: int



class _GroundedAnswer(BaseModel):
    """
    Structured output schema to enforce format and grounding.
    The model must cite excerpt numbers like [1] within each bullet.
    """

    insufficient: bool = False
    answer_bullets: List[str] = Field(default_factory=list)
    clarifying_questions: List[str] = Field(default_factory=list)


def _load_tables() -> List[Dict[str, Any]]:
    if not os.path.exists(TABLES_PATH):
        raise FileNotFoundError(f"Missing file: {TABLES_PATH}")
    with open(TABLES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return list(raw.get("tables") or [])

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
def script_js() -> FileResponse:
    """
    Serve the real JS file under frontend/ so you can edit it directly.

    The frontend JS calls:
      - POST /api/ask
      - GET /api/tables
      - GET /api/tables/{table_id}
    """
    js_path = os.path.join(FRONTEND_DIR, "script.js")
    if not os.path.exists(js_path):
        raise HTTPException(status_code=404, detail="frontend/script.js not found")
    return FileResponse(js_path, media_type="text/javascript; charset=utf-8")


@app.get("/health")
def health() -> Dict[str, Any]:
    # Helpful for quick sanity checks.
    chunks = retrieve_chunks("Find the 5 most important things in the PGP Document", k=5)
    return {
        "ok": True,
        "heading_chunks": len(chunks),
        "tables": len(_tables_by_id()),
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        chunks = retrieve_chunks(req.question, k=req.top_k)
    except Exception as e:
        # Make missing API key errors much easier to understand in the UI.
        try:
            import openai  # type: ignore

            if isinstance(e, openai.AuthenticationError):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "OpenAI authentication failed. Set OPENAI_API_KEY in .env "
                        "and restart the server."
                    ),
                ) from e
        except Exception:
            # If OpenAI isn't installed or types differ, fall through.
            pass
        raise

    results: List[AskResult] = [
        AskResult(
            content=c.page_content,
            section_index=c.metadata["section_index"],
            score=1.0,
            source=c.metadata["source"],
            snippet=c.page_content,
        )
        for c in chunks
    ]
    return AskResponse(question=req.question, top_k=req.top_k, results=results)


def _build_excerpt_block(chunks: List[Document]) -> Tuple[str, List[ChatCitation]]:
    citations: List[ChatCitation] = []
    parts: List[str] = []
    for i, c in enumerate(chunks, start=1):
        meta = c.metadata or {}
        section_index = int(meta.get("section_index") or 0)
        heading_title = meta.get("heading_title") or "Unknown heading"
        source = str(meta.get("source") or "")
        content = str(c.page_content or "")
        citations.append(ChatCitation(ref=i, section_index=section_index, source=source, heading_title=heading_title))
        parts.append(
            f"[{i}] {heading_title}; source: {source})\n{content}".strip()
        )
    return "\n\n".join(parts).strip(), citations


def _get_chat_model() -> ChatOpenAI:
    model_name = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    # Keep temperature low for faithful, excerpt-grounded answers.
    return ChatOpenAI(model=model_name, temperature=0)


_CITATION_RE = re.compile(r"\[(\d+)\]")


def _format_grounded_answer(payload: _GroundedAnswer) -> str:
    if payload.insufficient or not payload.answer_bullets:
        questions = [q.strip() for q in (payload.clarifying_questions or []) if q.strip()]
        if questions:
            qs = "\n".join(f"- {q}" for q in questions[:2])
            return "Answer: I don’t have enough in the provided excerpts to answer.\n\nClarifying questions:\n" + qs + "\n\nCitations: none"
        return "Answer: I don’t have enough in the provided excerpts to answer.\n\nCitations: none"

    def _bullet_text(s: str) -> str:
        t = s.strip()
        # Avoid double bullets: LLM often returns "- item; [1]" but we add "- " when formatting.
        if t.startswith("-"):
            t = t.lstrip("-").strip()
        return t

    bullets = [_bullet_text(b) for b in payload.answer_bullets if b and b.strip()]
    bullets = [b for b in bullets if b][:8]
    if not bullets:
        return "Answer: I don’t have enough in the provided excerpts to answer.\n\nCitations: none"

    # Enforce: every bullet must include at least one [n] citation.
    cleaned: List[str] = []
    for b in bullets:
        if not _CITATION_RE.search(b):
            # If the model forgot citations, degrade safely instead of hallucinating.
            continue
        cleaned.append(b)

    if not cleaned:
        return "Answer: I don’t have enough in the provided excerpts to answer.\n\nCitations: none"

    used = sorted({int(m.group(1)) for b in cleaned for m in _CITATION_RE.finditer(b)})
    used_str = ", ".join(f"[{n}]" for n in used) if used else "none"
    out = "Answer:\n" + "\n".join(f"- {b}" for b in cleaned) + f"\n\nCitations: {used_str}"
    return out.strip()


chat_history = []

@app.post("/api/clear_chat_history")
async def clear_chat_history(request: Request):
    global chat_history
    chat_history = []
    return {"ok": True, "message": "Chat history cleared."}

def _answer_with_llm(question: str, excerpts: str) -> str:
    # Agent loop: the LLM may request more excerpts by calling the
    # `retrieve_excerpts` tool (implemented below). The loop allows up
    # to 5 tool calls to gather evidence before producing a final
    # grounded answer using the structured output schema.
    llm = _get_chat_model()

    # Aggregate excerpts returned by tool calls (start with provided excerpts)
    aggregated_excerpts = excerpts or ""

    # Keep a short history for the LLM to reference
    messages = [("system", SYSTEM_PROMPT)]
    messages.extend(chat_history)

    # Produce final structured grounded answer using all aggregated excerpts
    user_msg = (
        "User question:\n"
        f"{question}\n\n"
        "Excerpts (numbered for citation):\n"
        f"{aggregated_excerpts}\n"
    )
    
    # print(aggregated_excerpts)
    structured = llm.with_structured_output(_GroundedAnswer)
    payload = structured.invoke([("system", SYSTEM_PROMPT), ("human", user_msg)])
    if isinstance(payload, dict):
        payload = _GroundedAnswer(**payload)

    answer = _format_grounded_answer(payload)
    chat_history.append(("human", question))
    chat_history.append(("assistant", answer))
    return answer

def query_modified_get_chunks(question: str, k: int = 10):
    if MODIFY_QUERY:
        llm = _get_chat_model()
        clarify_prompt = (
            "If the question is asking for something that requires information from before, then rewrite the question to address what was asked from before."
            "You are looking at the EPA PGP (Pesticide General Permit) Legal Document. Your new question should not contain the words EPA Pesticide General Permit or similar."
            "If the chat history is not empty, and the question refers to it, then modify. Otherwise, DO NOT MODIFY THE QUESTION AT ALL"
            "Keep modified responses under 20 words."
        )
        messages = [("system", clarify_prompt)]
        messages.extend(chat_history)
        print(chat_history)
        messages.append(("human", f"Question: {question}"))
        clarified_question = llm.invoke(messages).content.strip()
        # Fallback if LLM fails
        if not clarified_question:
            clarified_question = question
    else:
        clarified_question = question

    print("Clarifying question: ", clarified_question)
    chunks = retrieve_chunks(clarified_question, k=k)
    return chunks, clarified_question


def retrieve_excerpts(question: str, k: int = 10):
    """Tool wrapper the agent can call: returns (excerpts:str, citations:List[ChatCitation], chunks:List[Document])"""
    chunks, clarified = query_modified_get_chunks(question, k=k)
    excerpts, citations = _build_excerpt_block(list(chunks))
    return excerpts, citations, chunks


@app.post("/api/chat", response_model=ChatResponse)

def chat(req: ChatRequest) -> ChatResponse:
    try:
        chunks, clarified_question = query_modified_get_chunks(req.question, k=req.top_k)
        print("Clarifying question: ", clarified_question)
    except Exception as e:
        try:
            import openai  # type: ignore
            if isinstance(e, openai.AuthenticationError):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "OpenAI authentication failed. Set OPENAI_API_KEY in .env "
                        "and restart the server."
                    ),
                ) from e
        except Exception:
            pass
        raise

    excerpts, citations = _build_excerpt_block(list(chunks))
    if not excerpts:
        return ChatResponse(
            question=req.question,
            answer="Answer: I couldn't find any relevant excerpts.\nCitations: none",
            citations=[],
            retrieved_sections=[],
            top_k=req.top_k,
        )

    answer = _answer_with_llm(req.question, excerpts)

    retrieved_sections = [
        RetrievedSection(
            section_index=int(c.metadata.get("section_index", 0)),
            source=str(c.metadata.get("source", "")),
            heading_title=str(c.metadata.get("heading_title", "")),
            content=c.page_content,
        )
        for c in chunks[: req.top_k]
    ]

    return ChatResponse(
        question=req.question,
        answer=answer,
        citations=citations,
        retrieved_sections=retrieved_sections,
        top_k=req.top_k,
    )


@app.get("/api/tables")
def list_tables(query: Optional[str] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    return _list_tables_impl(query=query, limit=limit, offset=offset)


@app.get("/api/tables/{table_id}")
def get_table(table_id: str) -> Dict[str, Any]:
    return _get_table_impl(table_id)


def _list_tables_impl(query: Optional[str], limit: int, offset: int) -> Dict[str, Any]:
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


def _get_table_impl(table_id: str) -> Dict[str, Any]:
    table = _tables_by_id().get(table_id)
    if not table:
        raise HTTPException(status_code=404, detail=f"Table not found: {table_id}")
    return table


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=os.environ.get("HOST", APP_HOST),
        port=int(os.environ.get("PORT", APP_PORT)),
        reload=True,
    )

