// Chat history array to store messages (if needed for future use)
let chat_history = [];

// Clear chat history on page reload
window.onload = async function() {
  chat_history = [];
  // Optionally clear chat UI as well
  const chatContainer = document.getElementById("chat-container");
  if (chatContainer) chatContainer.innerHTML = "";
  // Clear server-side chat history
  try {
    await fetch("/api/clear_chat_history", { method: "POST" });
  } catch (e) {
    // Ignore errors
  }
};
function _escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Turn text with "- item" lines into HTML with real <ul>/<li> bullets. */
function _textToHtmlWithBullets(text) {
  if (!text || !text.trim()) return "";
  const lines = text.split("\n");
  let html = "";
  let inList = false;
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("- ")) {
      const bulletContent = trimmed.slice(2);
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += "<li>" + _escapeHtml(bulletContent) + "</li>";
    } else {
      if (inList) {
        html += "</ul>";
        inList = false;
      }
      html += _escapeHtml(line) + "<br/>";
    }
  }
  if (inList) html += "</ul>";
  return html;
}

function _appendMessage(text, role) {
  const chatContainer = document.getElementById("chat-container");
  const el = document.createElement("div");
  el.classList.add("message", role);
  el.innerText = text;
  chatContainer.appendChild(el);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return el;
}

/** Set message content, rendering lines that start with "- " as real bullet lists. */
function _setMessageContent(el, text) {
  const hasBullets = /(^|\n)-\s+/.test(text || "");
  if (hasBullets) {
    el.innerHTML = _textToHtmlWithBullets(text);
  } else {
    el.innerText = text;
  }
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
  if (!data) return "No response.";

  // New: prefer the LLM answer if present
  if (typeof data.answer === "string" && data.answer.trim() !== "") {
    return data.answer.trim();
  }

  // Fallback: show retrieval-only view
  if (!Array.isArray(data.results) || data.results.length === 0) {
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

function _formatChatResponse(data) {
  if (!data) return "No response.";
  const answer = (data.answer || "").trim();
  const citations = Array.isArray(data.citations) ? data.citations : [];
  let out = "";
  out += answer ? answer : "No answer returned.";
  if (citations.length) {
    out += "\n\nSources:\n";
    // Show up to 8 citations to keep UI readable.
    for (let i = 0; i < Math.min(8, citations.length); i++) {
      const c = citations[i];
      out += `- [${c.ref}] ${c.heading_title} (${c.source || "unknown"})\n`;
    }
  }
  out += "\n\nTip: use /ask <question> to see raw retrieved chunks.";
  out += "\nTip: use /tables [query] to browse extracted tables.";
  return out.trim();
}

function _formatTablesList(data) {
  const tables = data && data.tables ? data.tables : [];
  if (!tables.length) return "No tables matched.";
  let out = "Tables (showing up to your limit):\n";
  for (const t of tables) {
    const header = t.headers && t.headers.length ? t.headers[0] : "(no header)";
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

// Modal helper for showing retrieved evidence 
function _createEvidenceModal(sections) {
  const modal = document.createElement("div");
  modal.className = "modal";

  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";

  const box = document.createElement("div");
  box.className = "modal-box";

  const closeBtn = document.createElement("button");
  closeBtn.innerText = "Close";
  closeBtn.onclick = () => modal.remove();

  const title = document.createElement("h3");
  title.innerText = "Retrieved Evidence";

  box.appendChild(title);

  sections.forEach((s, i) => {
    const sec = document.createElement("div");
    sec.className = "evidence-card evidence-card--collapsed";
    const header = document.createElement("div");
    header.className = "evidence-card-header";
    header.innerHTML = `<strong>Section ${s.section_index}</strong> &middot; <em>${_escapeHtml(s.source || "")}</em>`;
    const body = document.createElement("div");
    body.className = "evidence-card-body";
    body.innerHTML = `<p>${_escapeHtml(s.content || "")}</p>`;
    header.addEventListener("click", () => {
      sec.classList.toggle("evidence-card--collapsed");
    });
    sec.appendChild(header);
    sec.appendChild(body);
    box.appendChild(sec);
  });

  box.appendChild(closeBtn);
  modal.appendChild(overlay);
  modal.appendChild(box);

  overlay.onclick = () => modal.remove();

  document.body.appendChild(modal);
}


async function sendMessage() {
  const input = document.getElementById("user-input");
  const message = (input.value || "").trim();
  if (message === "") return;

  _appendMessage(message, "user");
  const agentEl = _appendMessage("Thinking…", "agent");
  input.value = "";

  try {
    // /ask <question> (debug: show retrieved chunks)
    if (message.toLowerCase().startsWith("/ask")) {
      const q = message.slice("/ask".length).trim();
      if (!q) {
        agentEl.innerText = "Usage: /ask <question>";
        return;
      }
      const data = await _postJson("/api/ask", { question: q, top_k: 5 });
      _setMessageContent(agentEl, _formatAskResponse(data));
      return;
    }

    // /tables [query]
    if (message.toLowerCase().startsWith("/tables")) {
      const q = message.slice("/tables".length).trim();
      const url = "/api/tables?limit=10" + (q ? `&query=${encodeURIComponent(q)}` : "");
      const data = await _getJson(url);
      _setMessageContent(agentEl, _formatTablesList(data));
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
      _setMessageContent(agentEl, _formatTableDetail(data));
      return;
    }

    // Default: chat (answer grounded in retrieved excerpts)
const data = await _postJson("/api/chat", { question: message, top_k: 5 });

// show the main answer
_setMessageContent(agentEl, _formatChatResponse(data));

// add "Show retrieved evidence" button if evidence exists
if (Array.isArray(data.retrieved_sections) && data.retrieved_sections.length) {
  const btn = document.createElement("button");
  btn.innerText = "Show retrieved evidence";
  btn.className = "evidence-btn";

  btn.onclick = () => {
    _createEvidenceModal(data.retrieved_sections);
  };

  agentEl.appendChild(document.createElement("br"));
  agentEl.appendChild(btn);
}

  } catch (err) {
    agentEl.innerText = `Error: ${err && err.message ? err.message : String(err)}`;
  }
}

document.getElementById("user-input").addEventListener("keydown", function (event) {
  if (event.key === "Enter") {
    sendMessage();
  }
});
