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
