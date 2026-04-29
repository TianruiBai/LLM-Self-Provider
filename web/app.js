// Self-hosted Provider — minimal Open-WebUI-style chat client.
// Pure browser JS, talks to the same FastAPI app that serves this page.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const STORAGE_KEY = "provider.conversations.v1";
const ACTIVE_KEY = "provider.activeId.v1";
const SETTINGS_KEY = "provider.settings.v1";
const THEME_KEY = "provider.theme.v1";
const PENDING_ATTACH = []; // image attachments for the next message
const PENDING_DOCS = [];   // [{name, ext, format, text, size}] document attachments

// ---------------- markdown ----------------
if (window.marked) {
  marked.setOptions({
    gfm: true,
    breaks: true,
    highlight: (code, lang) => {
      try {
        if (window.hljs && lang && hljs.getLanguage(lang)) {
          return hljs.highlight(code, { language: lang }).value;
        }
        if (window.hljs) return hljs.highlightAuto(code).value;
      } catch {/* ignore */}
      return code;
    },
  });
}
function renderMarkdown(text) {
  if (!text) return "";
  if (!window.marked) return escapeHtml(text);
  const html = marked.parse(text);
  // Allow KaTeX-generated classes/attributes through DOMPurify.
  return window.DOMPurify
    ? DOMPurify.sanitize(html, { ADD_ATTR: ["aria-hidden", "encoding"], ADD_TAGS: ["math", "semantics", "annotation", "mrow", "mi", "mo", "mn", "ms", "mtext", "msup", "msub", "mfrac", "msqrt", "mroot", "mtable", "mtr", "mtd", "mstyle", "mspace"] })
    : html;
}
function renderMath(root) {
  if (!window.renderMathInElement) return;
  try {
    window.renderMathInElement(root, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    });
  } catch { /* ignore */ }
}
// Compress markdown for tight previews: drop fenced code, big tables, runs
// of blank lines, and turn ATX headings into plain bold leadlines so the
// preview reads as a paragraph.
function stripPreviewMd(text, maxChars = 600) {
  if (!text) return "";
  let s = String(text);
  s = s.replace(/```[\s\S]*?```/g, " "); // fenced code
  s = s.replace(/<br\s*\/?>/gi, "\n");
  s = s.replace(/^[ \t]*#{1,6}[ \t]*(.+?)\s*#*\s*$/gm, "**$1**");
  s = s.replace(/^[ \t]*\|.*\|[ \t]*$/gm, ""); // table rows
  s = s.replace(/^[ \t]*[-:|][- :|]+[ \t]*$/gm, ""); // table sep / hr
  s = s.replace(/\n{3,}/g, "\n\n");
  s = s.trim();
  if (s.length > maxChars) s = s.slice(0, maxChars).replace(/\s+\S*$/, "") + "…";
  return s;
}

// Render a builtin-tool result preview into the activity feed. The gateway
// publishes a JSON-encoded preview (truncated) on bus.publish; we try to
// pretty-print common shapes (web_search, web_fetch, rag.query, etc.).
function formatToolResult(name, raw) {
  const text = String(raw ?? "");
  if (!text) return { html: "<em class='dim'>(empty)</em>", expand: false };
  let parsed = null;
  try { parsed = JSON.parse(text); } catch { /* not JSON */ }
  if (parsed == null) {
    return { html: `<pre>${escapeHtml(text)}</pre>`, expand: false };
  }
  // web_search → array of {title, url, snippet}
  if (Array.isArray(parsed?.results) && (name || "").includes("search")) {
    if (!parsed.results.length) {
      return { html: "<em class='dim'>no results</em>", expand: true };
    }
    const items = parsed.results.slice(0, 8).map((r, i) => {
      const title = escapeHtml(r.title || r.url || "(no title)");
      const url = r.url ? escapeAttr(r.url) : "";
      const snip = escapeHtml((r.snippet || r.content || "").slice(0, 280));
      return `<li class="act-search-hit">
        <div class="act-search-title">${i + 1}. ${url ? `<a href="${url}" target="_blank" rel="noopener">${title}</a>` : title}</div>
        ${url ? `<div class="act-search-url">${escapeHtml(r.url)}</div>` : ""}
        <div class="act-search-snip">${snip}</div>
      </li>`;
    }).join("");
    return { html: `<ol class="act-search-list">${items}</ol>`, expand: true };
  }
  // web_fetch → {url, title, content}
  if (parsed?.content && (parsed.url || (name || "").includes("fetch"))) {
    const url = parsed.url ? escapeAttr(parsed.url) : "";
    const title = escapeHtml(parsed.title || parsed.url || "");
    const body = stripPreviewMd(parsed.content || "", 1200);
    const md = renderMarkdown(body);
    return {
      html: `${url ? `<div class="act-search-url"><a href="${url}" target="_blank" rel="noopener">${title}</a></div>` : ""}<div class="markdown">${md}</div>`,
      expand: true,
    };
  }
  // RAG query
  if (Array.isArray(parsed?.hits)) {
    const items = parsed.hits.slice(0, 6).map((h) => {
      const title = escapeHtml(h.title || h.doc_id || h.id || "");
      const score = (typeof h.score === "number") ? h.score.toFixed(3) : "?";
      const snip = escapeHtml((h.text || h.preview || "").slice(0, 240));
      return `<li><strong>${title}</strong> <span class="dim">score ${score}</span><div>${snip}</div></li>`;
    }).join("");
    return { html: `<ul class="act-rag-mini">${items}</ul>`, expand: true };
  }
  // Fallback: pretty JSON
  return { html: `<pre>${escapeHtml(JSON.stringify(parsed, null, 2))}</pre>`, expand: false };
}
function enhanceCodeBlocks(root) {
  root.querySelectorAll("pre > code").forEach((codeEl) => {
    const pre = codeEl.parentElement;
    if (pre.dataset.enhanced) return;
    pre.dataset.enhanced = "1";
    if (window.hljs && !codeEl.classList.contains("hljs")) {
      try { hljs.highlightElement(codeEl); } catch { /* ignore */ }
    }
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "code-copy";
    btn.textContent = "Copy";
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      try {
        await navigator.clipboard.writeText(codeEl.innerText);
        btn.textContent = "Copied";
        setTimeout(() => (btn.textContent = "Copy"), 1200);
      } catch { btn.textContent = "Err"; }
    });
    pre.appendChild(btn);
  });
}

// ---------------- state ----------------
let state = {
  models: [],
  conversations: loadConversations(),
  activeId: localStorage.getItem(ACTIVE_KEY) || null,
  controller: null,
  streaming: false,
  settings: loadSettings(),
};

function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}
function saveConversations() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.conversations));
  if (state.activeId) localStorage.setItem(ACTIVE_KEY, state.activeId);
}
function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch { return {}; }
}
function saveSettings() { localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings)); }

function activeConv() {
  return state.conversations.find(c => c.id === state.activeId);
}
function newConv(model, opts = {}) {
  const c = {
    id: crypto.randomUUID(),
    title: opts.title || "New chat",
    model: model || (state.models[0]?.id ?? ""),
    messages: opts.messages ? opts.messages.map(m => ({ ...m, id: m.id || crypto.randomUUID() })) : [],
    parentId: opts.parentId || null,
    createdAt: Date.now(),
  };
  state.conversations.unshift(c);
  state.activeId = c.id;
  saveConversations();
  return c;
}

// ---------------- theme ----------------
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem(THEME_KEY, theme);
  const tt = $("#theme-toggle");
  if (tt) {
    // Material icon span if present, otherwise textContent fallback.
    const mi = tt.querySelector(".mi");
    if (mi) mi.textContent = theme === "light" ? "dark_mode" : "light_mode";
    else tt.textContent = theme === "light" ? "☾" : "☀";
  }
  // Switch hljs theme stylesheet
  const link = $("#hljs-theme");
  if (link) {
    link.href = theme === "light"
      ? "https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github.min.css"
      : "https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github-dark.min.css";
  }
}

// ---------------- bootstrap ----------------
async function bootstrap() {
  applyTheme(localStorage.getItem(THEME_KEY) || "dark");
  bindUi();
  await Promise.all([refreshHealth(), refreshModels()]);
  if (!state.conversations.length) newConv();
  if (!state.activeId || !activeConv()) state.activeId = state.conversations[0].id;
  renderConversations();
  renderMessages();
  syncTopbar();
  setInterval(refreshHealth, 5000);
}

async function refreshModels() {
  try {
    const r = await fetch("/v1/models");
    const j = await r.json();
    state.models = j.data || [];
    // Pull admin status to learn which models are missing weights / can be downloaded.
    try {
      const sr = await fetch("/admin/status");
      if (sr.ok) {
        const sj = await sr.json();
        state.modelAdmin = {};
        for (const m of (sj.models || [])) state.modelAdmin[m.id] = m;
      }
    } catch { /* non-fatal */ }
    const sel = $("#model-select");
    const prev = sel.value;
    sel.innerHTML = "";
    for (const m of state.models) {
      const opt = document.createElement("option");
      opt.value = m.id;
      const tag = m.kind === "embedding" ? " (embed)"
                : m.kind === "sub_agent" ? " (sub-agent)"
                : m.kind === "vision" ? " (multi-modal)"
                : "";
      const admin = state.modelAdmin && state.modelAdmin[m.id];
      const missing = admin && admin.path_exists === false;
      opt.textContent = `${m.id}${tag}${missing ? " — not downloaded" : ""}`;
      if (m.kind === "embedding" || m.kind === "vision") opt.disabled = true;
      sel.appendChild(opt);
    }
    if (prev) sel.value = prev;
    syncTopbar();
  } catch (e) {
    console.warn("models load failed", e);
  }
}

async function refreshHealth() {
  try {
    const r = await fetch("/health");
    const j = await r.json();
    const dot = $("#status-dot");
    const txt = $("#status-text");
    dot.classList.remove("bad");
    dot.classList.add("ok");
    const active = j.active_chat_model ? `active: ${j.active_chat_model}` : "no chat model loaded";
    const sub = j.sub_agent_ready ? "sub-agent ready" : "sub-agent offline";
    const embed = j.embedder_ready ? "embedder ready" : "embedder offline";
    txt.textContent = `${active} · ${sub} · ${embed}`;
    $("#eject-model").disabled = !j.active_chat_model;
  } catch (e) {
    $("#status-dot").classList.remove("ok");
    $("#status-dot").classList.add("bad");
    $("#status-text").textContent = "gateway unreachable";
  }
}

// ---------------- UI bindings ----------------
function bindUi() {
  $("#new-chat").addEventListener("click", () => {
    newConv($("#model-select").value);
    renderConversations();
    renderMessages();
  });

  $("#model-select").addEventListener("change", (e) => {
    const c = activeConv();
    if (c) { c.model = e.target.value; saveConversations(); }
    syncTopbar();
  });

  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });
  $("#prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  $("#stop").addEventListener("click", () => {
    if (state.controller) state.controller.abort();
  });

  $("#eject-model").addEventListener("click", async () => {
    const btn = $("#eject-model");
    btn.disabled = true;
    try {
      const r = await fetch("/admin/unload", { method: "POST" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
    } catch (e) {
      console.warn("eject failed", e);
    } finally {
      refreshHealth();
    }
  });

  const fetchBtn = document.getElementById("fetch-model");
  if (fetchBtn) fetchBtn.addEventListener("click", fetchActiveModel);

  // Tab switching (Chat / Models / Knowledge / Activity).
  document.querySelectorAll(".tabs .tab").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // Knowledge sub-views (cards / network / ingest / query).
  document.querySelectorAll(".kb-view-btn").forEach(btn => {
    btn.addEventListener("click", () => switchKbView(btn.dataset.kbview));
  });

  const drawerClose = document.getElementById("kb-drawer-close");
  if (drawerClose) drawerClose.addEventListener("click", closeKbDrawer);

  const netMode = document.getElementById("kb-net-mode");
  if (netMode) netMode.addEventListener("change", renderKnowledgeNetwork);

  const modelsRefresh = document.getElementById("models-refresh");
  if (modelsRefresh) modelsRefresh.addEventListener("click", () => { refreshModels(); refreshHealth(); });

  $("#theme-toggle").addEventListener("click", () => {
    const next = (document.documentElement.dataset.theme === "light") ? "dark" : "light";
    applyTheme(next);
  });

  $("#open-settings").addEventListener("click", () => openSettingsDialog());
  $("#settings-reset").addEventListener("click", () => {
    state.settings = {};
    saveSettings();
    openSettingsDialog();
  });
  $("#settings-dialog").addEventListener("close", () => readSettingsDialog());

  $("#attach-btn").addEventListener("click", () => $("#attach-image").click());
  $("#attach-image").addEventListener("change", onAttachImages);

  const docBtn = document.getElementById("attach-doc-btn");
  const docIn = document.getElementById("attach-doc");
  if (docBtn && docIn) {
    docBtn.addEventListener("click", () => docIn.click());
    docIn.addEventListener("change", onAttachDocs);
  }

  $("#mic-btn").addEventListener("click", toggleVoiceRecord);

  // Drag & drop on the textarea / composer.
  setupComposerDragDrop();

  // Platform detection + mobile sidebar drawer.
  setupPlatform();

  // Persisted toggle for the built-in web tools.
  const toolsEl = $("#tools-toggle");
  if (toolsEl) {
    toolsEl.checked = localStorage.getItem("provider.tools.v1") === "1";
    toolsEl.addEventListener("change", () => {
      localStorage.setItem("provider.tools.v1", toolsEl.checked ? "1" : "0");
    });
  }

  $("#ing-summarize").addEventListener("click", summarizeAndIngest);

  $("#kb-refresh").addEventListener("click", refreshKnowledgeCards);
  $("#kb-filter-source").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); refreshKnowledgeCards(); } });
  $("#kb-filter-tag").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); refreshKnowledgeCards(); } });

  $("#ing-submit").addEventListener("click", ingestSubmit);
  $("#ing-file").addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    const reads = await Promise.all(files.map(async (f) => ({
      text: await f.text(),
      metadata: { id: f.name, title: f.name },
    })));
    // Append into the textarea as a JSON-marker that ingest will pick up.
    window.__pendingFileDocs = reads;
    $("#ing-status").textContent = `${reads.length} file(s) ready to ingest.`;
  });
  $("#rq-submit").addEventListener("click", ragQuery);
}

function syncTopbar() {
  const c = activeConv();
  if (c && c.model) $("#model-select").value = c.model;
  // Toggle the "Download" button based on the *currently selected* model's
  // admin status (path_exists + download spec). Any in-progress download is
  // shown via the progress bar driven by SSE.
  const sel = $("#model-select");
  const id = sel ? sel.value : "";
  const admin = (state.modelAdmin || {})[id];
  const btn = document.getElementById("fetch-model");
  if (!btn) return;
  const dlInProgress = state.downloads && state.downloads[id]
    && !["done", "error", "skip"].includes(state.downloads[id].phase);
  if (dlInProgress) {
    btn.hidden = false;
    btn.disabled = true;
    btn.textContent = "⬇ Downloading…";
  } else if (admin && admin.path_exists === false && admin.download) {
    btn.hidden = false;
    btn.disabled = false;
    btn.textContent = "⬇ Download model";
  } else {
    btn.hidden = true;
    btn.disabled = false;
  }
}

function fmtBytes(n) {
  if (!n && n !== 0) return "?";
  const u = ["B","KiB","MiB","GiB","TiB"];
  let i = 0; let v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${u[i]}`;
}

function applyDownloadProgress(modelId, ev) {
  state.downloads = state.downloads || {};
  state.downloads[modelId] = { ...(state.downloads[modelId] || {}), ...ev };
  const sel = $("#model-select");
  if (!sel || sel.value !== modelId) return;  // only render for the visible model
  const wrap = document.getElementById("fetch-progress");
  const fill = document.getElementById("fetch-bar-fill");
  const text = document.getElementById("fetch-progress-text");
  if (!wrap || !fill || !text) return;
  const phase = ev.phase;
  const done = phase === "done" || phase === "complete" || phase === "skip";
  const failed = phase === "error";
  if (failed) {
    wrap.hidden = false;
    fill.style.width = "100%";
    fill.classList.add("err");
    text.textContent = `error: ${ev.error || "failed"}`;
    syncTopbar();
    return;
  }
  if (done) {
    wrap.hidden = false;
    fill.classList.remove("err");
    fill.style.width = "100%";
    text.textContent = phase === "skip" ? "already on disk" : "✓ done";
    setTimeout(() => { wrap.hidden = true; }, 4000);
    // Refresh admin status so the dropdown clears the "not downloaded" tag.
    refreshModels();
    return;
  }
  if (phase === "progress" || phase === "begin" || phase === "start" || phase === "queued") {
    wrap.hidden = false;
    fill.classList.remove("err");
    const dl = ev.downloaded || 0;
    const total = ev.total || null;
    if (total) {
      const pct = Math.min(100, (dl / total) * 100);
      fill.style.width = pct.toFixed(1) + "%";
      text.textContent = `${pct.toFixed(0)}% · ${fmtBytes(dl)} / ${fmtBytes(total)}`;
    } else {
      // Indeterminate: pulse the bar.
      fill.style.width = "30%";
      text.textContent = `${fmtBytes(dl)} downloaded`;
    }
  }
}

async function fetchActiveModel() {
  const sel = $("#model-select");
  const id = sel && sel.value;
  if (!id) return;
  const btn = document.getElementById("fetch-model");
  btn.disabled = true;
  btn.textContent = "⬇ Starting…";
  try {
    const r = await fetch("/admin/fetch-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: id }),
    });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(`HTTP ${r.status}: ${t}`);
    }
  } catch (e) {
    console.warn("fetch-model failed", e);
    applyDownloadProgress(id, { phase: "error", error: String(e.message || e) });
  }
}

// ---------------- conversations ----------------
function renderConversations() {
  const ul = $("#conversations");
  ul.innerHTML = "";
  for (const c of state.conversations) {
    const li = document.createElement("li");
    li.dataset.id = c.id;
    if (c.id === state.activeId) li.classList.add("active");
    const title = document.createElement("span");
    title.className = "title";
    title.textContent = c.title || "(untitled)";
    const del = document.createElement("button");
    del.className = "del";
    del.textContent = "✕";
    del.title = "Delete conversation";
    del.addEventListener("click", (ev) => {
      ev.stopPropagation();
      state.conversations = state.conversations.filter(x => x.id !== c.id);
      if (state.activeId === c.id) state.activeId = state.conversations[0]?.id || null;
      if (!state.activeId) newConv();
      saveConversations();
      renderConversations();
      renderMessages();
    });
    li.appendChild(title);
    li.appendChild(del);
    li.addEventListener("click", () => {
      state.activeId = c.id;
      saveConversations();
      renderConversations();
      renderMessages();
      syncTopbar();
    });
    ul.appendChild(li);
  }
}

// ---------------- messages ----------------
function renderMessages() {
  const c = activeConv();
  const box = $("#messages");
  box.innerHTML = "";
  if (!c || !c.messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-hint";
    empty.innerHTML = `<div class="empty-title">Start chatting</div>
      <div>Pick a model above. The first message will load it on the GPU.</div>`;
    box.appendChild(empty);
    return;
  }
  for (const m of c.messages) appendMessageDom(m);
  scrollToBottom();
}

function appendMessageDom(m) {
  if (!m.id) m.id = crypto.randomUUID();
  const box = $("#messages");
  const wrap = document.createElement("div");
  wrap.className = `msg ${m.role}`;
  wrap.dataset.role = m.role;
  wrap.dataset.mid = m.id;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = m.role === "user" ? "U" : (m.role === "assistant" ? "A" : "S");
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  let reasoningEl = null;
  if (m.reasoning) {
    reasoningEl = document.createElement("details");
    reasoningEl.className = "reasoning";
    const summary = document.createElement("summary");
    summary.textContent = "Reasoning";
    reasoningEl.appendChild(summary);
    const r = document.createElement("div");
    r.className = "reasoning-body";
    r.textContent = m.reasoning;
    reasoningEl.appendChild(r);
    bubble.appendChild(reasoningEl);
  }

  // Render attached images (user multimodal messages).
  if (Array.isArray(m.images) && m.images.length) {
    const imgs = document.createElement("div");
    imgs.className = "msg-images";
    for (const url of m.images) {
      const img = document.createElement("img");
      img.src = url;
      imgs.appendChild(img);
    }
    bubble.appendChild(imgs);
  }

  // Render attached document pills (user message).
  if (Array.isArray(m.docs) && m.docs.length) {
    const dwrap = document.createElement("div");
    dwrap.className = "msg-docs";
    for (const d of m.docs) {
      const pill = document.createElement("span");
      pill.className = "msg-doc-pill" + (d.error ? " err" : "");
      const sz = d.size ? ` · ${fmtBytes(d.size)}` : "";
      pill.innerHTML = `<span class="ic">📄</span><span class="nm"></span><span class="meta"></span>`;
      pill.querySelector(".nm").textContent = d.name;
      pill.querySelector(".meta").textContent = `${d.format || ""}${sz}`;
      pill.title = d.error ? `error: ${d.error}` : `${d.name} (${d.format || "?"})`;
      dwrap.appendChild(pill);
    }
    bubble.appendChild(dwrap);
  }

  const body = document.createElement("div");
  body.className = "body markdown";
  const displayed = (m.role === "user" && m.displayContent != null) ? m.displayContent : (m.content || "");
  body.innerHTML = renderMarkdown(displayed);
  enhanceCodeBlocks(body);
  renderMath(body);
  bubble.appendChild(body);

  if (m.meta) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = m.meta;
    bubble.appendChild(meta);
  }

  // Per-message action toolbar.
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  actions.appendChild(makeActionBtn("⧉", "Copy", () => copyText(m.content || "")));
  if (m.role === "assistant") {
    actions.appendChild(makeActionBtn("⎘", "Branch from here", () => branchFromMessage(m.id)));
    actions.appendChild(makeActionBtn("⮶", "Summarize → RAG", () => summarizeMessageToRag(m)));
    actions.appendChild(makeActionBtn("↻", "Regenerate", () => regenerateFrom(m.id)));
  }
  if (m.role === "user") {
    actions.appendChild(makeActionBtn("✎", "Edit & resend", () => editMessage(m.id)));
  }
  actions.appendChild(makeActionBtn("✕", "Delete", () => deleteMessage(m.id)));
  bubble.appendChild(actions);

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  box.appendChild(wrap);
  return { wrap, bubble, body, reasoningEl };
}

function makeActionBtn(label, title, handler) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "msg-action";
  b.title = title;
  b.textContent = label;
  b.addEventListener("click", (e) => { e.stopPropagation(); handler(); });
  return b;
}

async function copyText(t) {
  try { await navigator.clipboard.writeText(t || ""); } catch { /* ignore */ }
}
function deleteMessage(mid) {
  const c = activeConv();
  if (!c) return;
  c.messages = c.messages.filter(x => x.id !== mid);
  saveConversations();
  renderMessages();
}
function editMessage(mid) {
  const c = activeConv();
  const m = c?.messages.find(x => x.id === mid);
  if (!m) return;
  $("#prompt").value = m.content || "";
  // Restore the message's images/documents into the pending tray so the
  // regenerated send re-attaches them automatically.
  restorePendingFromUserMsg(m);
  // Drop this user message and everything after it; the next send replays.
  const idx = c.messages.findIndex(x => x.id === mid);
  c.messages = c.messages.slice(0, idx);
  saveConversations();
  renderMessages();
  $("#prompt").focus();
}
function regenerateFrom(mid) {
  const c = activeConv();
  if (!c) return;
  const idx = c.messages.findIndex(x => x.id === mid);
  if (idx < 0) return;
  // Find the preceding user message; truncate to it, then resend.
  let userIdx = idx - 1;
  while (userIdx >= 0 && c.messages[userIdx].role !== "user") userIdx--;
  if (userIdx < 0) return;
  const userMsg = c.messages[userIdx];
  const userText = userMsg.displayContent || userMsg.content;
  // Re-populate PENDING_ATTACH / PENDING_DOCS so any image/document the user
  // originally attached survives the regenerate round-trip.
  restorePendingFromUserMsg(userMsg);
  c.messages = c.messages.slice(0, userIdx);
  saveConversations();
  renderMessages();
  $("#prompt").value = typeof userText === "string" ? userText : "";
  setTimeout(sendMessage, 0);
}

// Re-stage a previously sent user message's attachments into the composer
// trays so the next send rebuilds the same multimodal payload. Used by
// editMessage and regenerateFrom.
function restorePendingFromUserMsg(m) {
  if (!m || m.role !== "user") return;
  PENDING_ATTACH.length = 0;
  PENDING_DOCS.length = 0;
  if (Array.isArray(m.images)) {
    for (const url of m.images) {
      if (typeof url === "string" && url) PENDING_ATTACH.push(url);
    }
  }
  if (Array.isArray(m.docsPayload)) {
    for (const d of m.docsPayload) {
      if (!d || typeof d !== "object") continue;
      PENDING_DOCS.push({
        id: d.id,
        name: d.name || "document",
        ext: (d.format || "text"),
        format: d.format || "text",
        size: d.size || (d.text ? d.text.length : 0),
        text: d.text || "",
        uploading: false,
      });
    }
  }
  renderAttachments();
}
function branchFromMessage(mid) {
  const c = activeConv();
  if (!c) return;
  const idx = c.messages.findIndex(x => x.id === mid);
  if (idx < 0) return;
  const cloned = c.messages.slice(0, idx + 1).map(m => ({ ...m, id: crypto.randomUUID() }));
  newConv(c.model, {
    title: (c.title || "Branch") + " — branch",
    messages: cloned,
    parentId: c.id,
  });
  renderConversations();
  renderMessages();
  syncTopbar();
}
async function summarizeMessageToRag(m) {
  const text = (m.content || "").trim();
  if (!text) return;
  const title = prompt("Title for this knowledge card:", (activeConv()?.title || "summary").slice(0, 60));
  if (title === null) return;
  try {
    const r = await fetch("/admin/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, title: title || "summary", source: "summaries", tags: ["summary", "chat"], unload_after: true }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || JSON.stringify(j));
    alert(`Saved as knowledge card “${j.title}” (${j.ingest?.chunks ?? 0} chunks).`);
  } catch (e) {
    alert(`Summarize failed: ${e.message || e}`);
  }
}

function scrollToBottom() {
  const box = $("#messages");
  box.scrollTop = box.scrollHeight;
}

// ---------------- send ----------------
async function sendMessage() {
  if (state.streaming) return;
  const text = $("#prompt").value.trim();
  const hasDocs = PENDING_DOCS.length > 0;
  if (!text && !hasDocs && PENDING_ATTACH.length === 0) return;
  const c = activeConv() || newConv($("#model-select").value);
  c.model = $("#model-select").value || c.model;
  if (!c.model) { alert("Please pick a model."); return; }

  // Block sending while docs are still uploading.
  if (PENDING_DOCS.some(d => d.uploading)) {
    alert("Some documents are still being processed. Please wait.");
    return;
  }

  // Build the message content with attached document text prepended as a
  // labeled block; the visible bubble preserves the user's actual prompt.
  let composedContent = text;
  let docsForMsg = null;
  let docsForApi = null;
  const useToolsForBuild = $("#tools-toggle")?.checked === true;
  if (hasDocs) {
    const ready = PENDING_DOCS.filter(d => !d.error && d.text);
    if (useToolsForBuild && ready.length) {
      // Attach docs as structured payload — the server exposes
      // list_documents / read_document / search_documents so the model can
      // browse them on demand instead of inlining the entire text.
      docsForApi = ready.map((d, i) => ({
        id: d.id || `doc${i + 1}`,
        name: d.name,
        format: d.format || "text",
        size: d.size || (d.text ? d.text.length : 0),
        text: d.text,
      }));
      const listing = ready.map((d, i) => `- ${d.name} (${d.format || "text"}, ${d.text.length.toLocaleString()} chars)`).join("\n");
      const header = `Attached documents (use the document tools to read/search):\n${listing}`;
      composedContent = composedContent ? `${header}\n\n---\n\n${composedContent}` : header;
    } else {
      const blocks = ready.map(d => {
        const fence = "```";
        return `${fence} ${d.format || "text"} title=${JSON.stringify(d.name)}\n${d.text}\n${fence}`;
      });
      if (blocks.length) {
        const header = "Attached documents:\n\n" + blocks.join("\n\n");
        composedContent = composedContent ? `${header}\n\n---\n\n${composedContent}` : header;
      }
    }
    docsForMsg = PENDING_DOCS.map(d => ({ name: d.name, format: d.format, size: d.size, error: d.error || null }));
    PENDING_DOCS.length = 0;
  }

  const userMsg = { role: "user", content: composedContent, displayContent: text || "(documents attached)", id: crypto.randomUUID() };
  if (docsForMsg) userMsg.docs = docsForMsg;
  if (docsForApi) userMsg.docsPayload = docsForApi;
  if (PENDING_ATTACH.length) {
    userMsg.images = PENDING_ATTACH.slice();
    PENDING_ATTACH.length = 0;
    renderAttachments();
  } else {
    renderAttachments();
  }
  c.messages.push(userMsg);
  if (c.messages.length === 1 || c.title === "New chat") {
    c.title = (text || (docsForMsg && docsForMsg[0] && docsForMsg[0].name) || "Chat").slice(0, 40);
  }
  saveConversations();
  $("#prompt").value = "";
  renderConversations();
  appendMessageDom(userMsg);
  scrollToBottom();

  const useRag = $("#rag-toggle").checked;
  const stream = $("#stream-toggle").checked;
  const useTools = $("#tools-toggle")?.checked === true;
  const s = state.settings || {};

  // Build OpenAI-format messages with optional system + multimodal images.
  const apiMessages = [];
  if (s.system && s.system.trim()) apiMessages.push({ role: "system", content: s.system });
  for (const m of c.messages) {
    if (m.role === "user" && Array.isArray(m.images) && m.images.length) {
      const parts = [];
      if (m.content) parts.push({ type: "text", text: m.content });
      for (const url of m.images) parts.push({ type: "image_url", image_url: { url } });
      apiMessages.push({ role: "user", content: parts });
    } else {
      apiMessages.push({ role: m.role, content: m.content });
    }
  }

  const body = { model: c.model, messages: apiMessages, stream };
  if (Number.isFinite(s.temperature)) body.temperature = s.temperature;
  if (Number.isFinite(s.top_p)) body.top_p = s.top_p;
  if (Number.isFinite(s.top_k)) body.top_k = s.top_k;
  if (Number.isFinite(s.max_tokens)) body.max_tokens = s.max_tokens;
  if (Number.isFinite(s.repeat_penalty)) body.repeat_penalty = s.repeat_penalty;
  if (useRag) {
    body.rag = {
      enabled: true,
      top_k: parseInt($("#rag-topk").value || "4", 10),
    };
    const src = $("#rag-source").value.trim();
    if (src) body.rag.source = src;
  }
  if (useTools) {
    body.tools_builtin = true;
  }
  // Aggregate any document payloads attached to user messages in this
  // conversation so the doc tools can read them across follow-ups.
  const allDocs = [];
  for (const m of c.messages) {
    if (m.role === "user" && Array.isArray(m.docsPayload)) {
      for (const d of m.docsPayload) allDocs.push(d);
    }
  }
  if (allDocs.length) {
    body.documents = allDocs;
  }

  const asstMsg = { role: "assistant", content: "", reasoning: "", id: crypto.randomUUID() };
  c.messages.push(asstMsg);
  const dom = appendMessageDom(asstMsg);
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  dom.body.appendChild(cursor);

  state.controller = new AbortController();
  state.streaming = true;
  $("#send").disabled = true;
  $("#stop").hidden = false;
  $("#composer-meta").textContent = `→ ${c.model}${useRag ? " · RAG" : ""}${stream ? " · streaming" : ""}`;

  const t0 = performance.now();
  try {
    const r = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: state.controller.signal,
    });

    if (!stream) {
      const j = await r.json();
      if (!r.ok) throw new Error(j.error?.message || j.detail || JSON.stringify(j));
      const choice = j.choices?.[0]?.message || {};
      asstMsg.content = choice.content || "";
      if (choice.reasoning_content) asstMsg.reasoning = choice.reasoning_content;
      const u = j.usage || {};
      asstMsg.meta = `${j.model || c.model} · ${u.completion_tokens ?? "?"} tok` +
        (j.timings?.predicted_per_second ? ` · ${j.timings.predicted_per_second.toFixed(1)} tok/s` : "");
      cursor.remove();
      dom.body.innerHTML = renderMarkdown(asstMsg.content);
      enhanceCodeBlocks(dom.body);
      renderMath(dom.body);
      const mdiv = document.createElement("div");
      mdiv.className = "meta";
      mdiv.textContent = asstMsg.meta;
      dom.bubble.appendChild(mdiv);
    } else {
      // SSE stream — robust parser: events end with \n\n; each event has
      // one or more lines, only `data:` carries JSON. Fragments arrive
      // unaligned across reads, so we accumulate and split.
      if (!r.ok || !r.body) {
        const txt = await r.text();
        throw new Error(`HTTP ${r.status}: ${txt}`);
      }
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let lastModel = c.model;
      let outTokens = 0;
      let lastUsage = null;
      let firstTokenAt = null;
      let reasoningPre = null;
      let pendingRender = false;
      const scheduleRender = () => {
        if (pendingRender) return;
        pendingRender = true;
        requestAnimationFrame(() => {
          pendingRender = false;
          dom.body.innerHTML = renderMarkdown(asstMsg.content);
          enhanceCodeBlocks(dom.body);
          dom.body.appendChild(cursor);
        });
      };
      const ensureReasoningPre = () => {
        if (reasoningPre) return reasoningPre;
        const det = document.createElement("details");
        det.className = "reasoning";
        det.open = true;
        const sm = document.createElement("summary");
        sm.textContent = "Reasoning";
        det.appendChild(sm);
        reasoningPre = document.createElement("div");
        reasoningPre.className = "reasoning-body";
        det.appendChild(reasoningPre);
        dom.bubble.insertBefore(det, dom.body);
        return reasoningPre;
      };

      const handleEvent = (rawEvent) => {
        const lines = rawEvent.split(/\r?\n/);
        const dataParts = [];
        for (const line of lines) {
          if (line.startsWith(":")) continue;
          if (!line.startsWith("data:")) continue;
          dataParts.push(line.slice(5).replace(/^ /, ""));
        }
        if (!dataParts.length) return false;
        const data = dataParts.join("\n");
        if (data === "[DONE]") return true;
        let j;
        try { j = JSON.parse(data); } catch { return false; }
        if (j.error) {
          asstMsg.content += `\n⚠ ${j.error.message || JSON.stringify(j.error)}`;
          scheduleRender();
          return false;
        }
        if (j.model) lastModel = j.model;
        if (j.usage) lastUsage = j.usage;
        const delta = j.choices?.[0]?.delta || {};
        if (delta.reasoning_content) {
          asstMsg.reasoning += delta.reasoning_content;
          ensureReasoningPre().textContent = asstMsg.reasoning;
        }
        if (delta.content) {
          if (firstTokenAt === null) firstTokenAt = performance.now();
          asstMsg.content += delta.content;
          outTokens++;
          scheduleRender();
          const elapsed = (performance.now() - firstTokenAt) / 1000;
          if (elapsed > 0.25) {
            const tps = outTokens / elapsed;
            $("#composer-meta").textContent =
              `→ ${lastModel}${useRag ? " · RAG" : ""} · ${outTokens} chunks · ${tps.toFixed(1)}/s`;
          }
        }
        return false;
      };

      let done = false;
      while (!done) {
        const chunk = await reader.read();
        if (chunk.done) {
          if (buf.trim()) handleEvent(buf);
          break;
        }
        buf += dec.decode(chunk.value, { stream: true });
        let idx;
        while ((idx = buf.search(/\r?\n\r?\n/)) !== -1) {
          const sepLen = buf.slice(idx).match(/^\r?\n\r?\n/)[0].length;
          const evt = buf.slice(0, idx);
          buf = buf.slice(idx + sepLen);
          if (handleEvent(evt)) { done = true; break; }
        }
        scrollToBottom();
      }
      cursor.remove();
      // final render without cursor
      dom.body.innerHTML = renderMarkdown(asstMsg.content);
      enhanceCodeBlocks(dom.body);
      renderMath(dom.body);
      const dt = (performance.now() - t0) / 1000;
      const tokDelta = lastUsage?.completion_tokens ?? outTokens;
      const tps = lastUsage?.predicted_per_second ?? (firstTokenAt ? outTokens / ((performance.now() - firstTokenAt) / 1000) : null);
      asstMsg.meta = `${lastModel} · ${tokDelta} tok · ${dt.toFixed(2)}s${tps ? ` · ${tps.toFixed(1)} tok/s` : ""}`;
      const mdiv = document.createElement("div");
      mdiv.className = "meta";
      mdiv.textContent = asstMsg.meta;
      dom.bubble.appendChild(mdiv);
    }
  } catch (e) {
    cursor.remove();
    asstMsg.content = `⚠ ${e.message || e}`;
    dom.body.innerHTML = renderMarkdown(asstMsg.content);
  } finally {
    saveConversations();
    state.streaming = false;
    state.controller = null;
    $("#send").disabled = false;
    $("#stop").hidden = true;
    $("#composer-meta").textContent = "";
    refreshHealth();
  }
}

// ---------------- RAG dialog ----------------
async function ingestSubmit() {
  const status = $("#ing-status");
  status.textContent = "ingesting…";
  const source = $("#ing-source").value.trim() || "docs";
  const tags = $("#ing-tags").value.split(",").map(s => s.trim()).filter(Boolean);
  const docs = [];
  const text = $("#ing-text").value.trim();
  if (text) {
    const title = $("#ing-title").value.trim();
    docs.push({ text, metadata: title ? { id: title, title } : {} });
  }
  if (window.__pendingFileDocs) {
    for (const d of window.__pendingFileDocs) docs.push(d);
  }
  if (!docs.length) { status.textContent = "Nothing to ingest."; return; }
  try {
    const r = await fetch("/rag/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ documents: docs, source, tags }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || JSON.stringify(j));
    status.textContent = `inserted ${j.inserted}, chunks ${j.chunks}`;
    $("#ing-text").value = "";
    $("#ing-file").value = "";
    window.__pendingFileDocs = null;
  } catch (e) {
    status.textContent = `error: ${e.message || e}`;
  }
}

async function ragQuery() {
  const text = $("#rq-text").value.trim();
  const top_k = parseInt($("#rq-topk").value || "5", 10);
  const source = $("#rq-source").value.trim() || undefined;
  const out = $("#rq-results");
  out.innerHTML = "searching…";
  try {
    const r = await fetch("/rag/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, top_k, source }),
    });
    const j = await r.json();
    out.innerHTML = "";
    if (!j.hits?.length) {
      out.textContent = "No hits.";
      return;
    }
    for (const h of j.hits) {
      const div = document.createElement("div");
      div.className = "hit";
      const meta = h.metadata || {};
      const title = meta.title || meta.id || h.id;
      div.innerHTML = `<div class="score">score ${h.score.toFixed(3)} · ${title}</div>
        <div class="text"></div>`;
      div.querySelector(".text").textContent = h.text;
      out.appendChild(div);
    }
  } catch (e) {
    out.textContent = `error: ${e.message || e}`;
  }
}

// ---------------- live activity (SSE /events) ----------------
const activity = {
  byId: new Map(),       // request id -> { card, contentEl, reasoningEl, ev }
  liveCount: 0,
  total: 0,
  errors: 0,
  tpsSamples: [],
  modelSet: new Set(),
  evtSource: null,
};

function bindActivityUi() {
  const baseEl = document.getElementById("activity-base");
  const epEl = document.getElementById("activity-endpoint");
  const url = `${location.protocol}//${location.host}/v1`;
  if (baseEl) baseEl.textContent = url;
  if (epEl) epEl.textContent = `endpoint: ${url}`;

  const clearBtn = document.getElementById("activity-clear");
  if (clearBtn) clearBtn.addEventListener("click", () => {
    document.getElementById("activity-list").innerHTML = "";
    activity.byId.clear();
    activity.total = 0;
    activity.errors = 0;
    activity.tpsSamples = [];
    activity.modelSet = new Set();
    refreshActivityCount();
    document.getElementById("activity-empty").hidden = false;
    refreshActivityStats();
    refreshActivityModelFilter();
  });

  // Filter inputs.
  ["act-filter-text", "act-filter-status", "act-filter-model"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("input", applyActivityFilters);
    if (el) el.addEventListener("change", applyActivityFilters);
  });
}

function refreshActivityCount() {
  const el = document.getElementById("act-count");
  const dot = document.getElementById("act-dot");
  if (!el || !dot) return;
  el.textContent = String(activity.liveCount || activity.total);
  el.classList.toggle("live", activity.liveCount > 0);
  dot.classList.remove("ok", "live", "bad");
  if (!activity.evtSource || activity.evtSource.readyState === 2) dot.classList.add("bad");
  else if (activity.liveCount > 0) dot.classList.add("live");
  else dot.classList.add("ok");
}

function connectActivity() {
  if (activity.evtSource && activity.evtSource.readyState !== 2) return;
  try {
    const es = new EventSource("/events");
    activity.evtSource = es;
    es.onopen = () => refreshActivityCount();
    es.onerror = () => refreshActivityCount();
    es.onmessage = (msg) => {
      let ev;
      try { ev = JSON.parse(msg.data); } catch { return; }
      handleActivityEvent(ev);
    };
  } catch (e) { console.warn("EventSource failed", e); }
}

function handleActivityEvent(ev) {
  const list = document.getElementById("activity-list");
  const empty = document.getElementById("activity-empty");
  if (ev.type === "model.download") {
    applyDownloadProgress(ev.model, ev);
    return;
  }
  if (ev.type === "request.start") {
    if (empty) empty.hidden = true;
    activity.liveCount++;
    activity.total++;
    if (ev.model) activity.modelSet.add(ev.model);
    refreshActivityCount();
    refreshActivityStats();
    refreshActivityModelFilter();
    const card = document.createElement("li");
    card.className = "act-card live";
    card.dataset.id = ev.id;
    card.dataset.status = "live";
    card.dataset.model = ev.model || "";
    card.dataset.client = ev.client || "";
    card.dataset.preview = (ev.preview || "").slice(0, 200);
    const ua = (ev.user_agent || "").slice(0, 60);
    card.innerHTML = `
      <div class="act-card-row">
        <span class="pill live">LIVE</span>
        <span class="model"></span>
        <span class="pill" data-path></span>
        ${ev.rag ? '<span class="pill rag">RAG</span>' : ""}
        ${ev.tools_builtin ? '<span class="pill tools">tools</span>' : ""}
        ${ev.stream ? '<span class="pill">stream</span>' : '<span class="pill">non-stream</span>'}
      </div>
      <div class="act-card-row"><span class="client"></span> <span class="footer ua"></span></div>
      <div class="preview"></div>
      <details class="act-section act-rag" hidden><summary>Knowledge base hits <span class="count"></span></summary><div class="act-rag-list"></div></details>
      <details class="act-section act-tools" hidden><summary>Tool calls <span class="count"></span></summary><div class="act-tools-list"></div></details>
      <details class="act-section act-reasoning" hidden><summary>Reasoning <span class="count"></span></summary><div class="stream-out reasoning markdown"></div></details>
      <details class="act-section act-content" hidden open><summary>Streamed output <span class="count"></span></summary><div class="stream-out content markdown"></div></details>
      <div class="footer status">streaming…</div>
    `;
    card.querySelector(".model").textContent = ev.model || "?";
    card.querySelector("[data-path]").textContent = ev.path || "";
    card.querySelector(".client").textContent = ev.client || "?";
    card.querySelector(".ua").textContent = ua;
    card.querySelector(".preview").textContent = ev.preview || "(no user message)";
    list.prepend(card);
    activity.byId.set(ev.id, {
      card,
      content: card.querySelector(".stream-out.content"),
      reasoning: card.querySelector(".stream-out.reasoning"),
      contentSection: card.querySelector(".act-content"),
      reasoningSection: card.querySelector(".act-reasoning"),
      contentCount: card.querySelector(".act-content .count"),
      reasoningCount: card.querySelector(".act-reasoning .count"),
      ragSection: card.querySelector(".act-rag"),
      ragList: card.querySelector(".act-rag-list"),
      ragCount: card.querySelector(".act-rag .count"),
      toolsSection: card.querySelector(".act-tools"),
      toolsList: card.querySelector(".act-tools-list"),
      toolsCount: card.querySelector(".act-tools .count"),
      contentText: "",
      reasoningText: "",
      toolsBuf: new Map(),
      toolsCounter: 0,
      startedAt: Date.now(),
    });
    return;
  }
  if (ev.type === "rag.hits") {
    const ent = activity.byId.get(ev.id);
    if (!ent) return;
    const hits = Array.isArray(ev.hits) ? ev.hits : [];
    if (!hits.length) return;
    ent.ragSection.hidden = false;
    ent.ragSection.open = true;
    ent.ragCount.textContent = `(${hits.length})`;
    ent.ragList.innerHTML = "";
    for (const h of hits) {
      const row = document.createElement("div");
      row.className = "act-rag-row";
      const score = (typeof h.score === "number") ? h.score.toFixed(3) : "?";
      row.innerHTML = `
        <div class="act-rag-head">
          <span class="act-rag-title"></span>
          <span class="act-rag-score"></span>
        </div>
        <div class="act-rag-meta"><span class="act-rag-src"></span></div>
        <div class="act-rag-prev markdown"></div>
      `;
      row.querySelector(".act-rag-title").textContent = h.title || h.doc_id || h.id || "(untitled)";
      row.querySelector(".act-rag-score").textContent = `score ${score}`;
      row.querySelector(".act-rag-src").textContent = `${h.source || "?"} · ${h.doc_id || ""}`;
      const prev = row.querySelector(".act-rag-prev");
      prev.innerHTML = renderMarkdown(stripPreviewMd(h.preview || "", 240));
      ent.ragList.appendChild(row);
    }
    return;
  }
  if (ev.type === "tool.call") {
    const ent = activity.byId.get(ev.id);
    if (!ent) return;
    ent.toolsSection.hidden = false;
    ent.toolsSection.open = true;
    ent.toolsCounter += 1;
    const idx = ent.toolsCounter;
    const row = document.createElement("div");
    row.className = "act-tool-row pending";
    row.innerHTML = `
      <div class="act-tool-head">
        <span class="mi">build</span>
        <span class="act-tool-name"></span>
        <span class="act-tool-status">running…</span>
      </div>
      <details class="act-tool-args"><summary>arguments</summary><pre></pre></details>
      <details class="act-tool-result" hidden><summary>result</summary><div class="markdown"></div></details>
    `;
    row.querySelector(".act-tool-name").textContent = ev.name || "(tool)";
    row.querySelector(".act-tool-args pre").textContent = ev.args_preview || "";
    ent.toolsList.appendChild(row);
    ent.toolsBuf.set(`${ev.name}|${idx}`, row);
    // Also remember by name for the next matching result.
    const queue = ent.toolsBuf.get(`pending:${ev.name}`) || [];
    queue.push(row);
    ent.toolsBuf.set(`pending:${ev.name}`, queue);
    ent.toolsCount.textContent = `(${ent.toolsList.children.length})`;
    return;
  }
  if (ev.type === "tool.result") {
    const ent = activity.byId.get(ev.id);
    if (!ent) return;
    const queue = ent.toolsBuf.get(`pending:${ev.name}`) || [];
    const row = queue.shift();
    if (!row) return;
    ent.toolsBuf.set(`pending:${ev.name}`, queue);
    row.classList.remove("pending");
    row.classList.add("done");
    const status = row.querySelector(".act-tool-status");
    if (status) status.textContent = "done";
    const resDet = row.querySelector(".act-tool-result");
    const resBox = resDet.querySelector(".markdown");
    resDet.hidden = false;
    const formatted = formatToolResult(ev.name, ev.result_preview);
    resBox.innerHTML = formatted.html;
    if (formatted.expand) resDet.open = true;
    enhanceCodeBlocks(resBox);
    return;
  }
  if (ev.type === "delta") {
    const ent = activity.byId.get(ev.id);
    if (!ent) return;
    const now = performance.now();
    if (ev.content) {
      ent.contentText += ev.content;
      ent.contentSection.hidden = false;
      // Throttle markdown re-render to once per animation frame.
      if (!ent._pendingContent) {
        ent._pendingContent = true;
        requestAnimationFrame(() => {
          ent._pendingContent = false;
          ent.content.innerHTML = renderMarkdown(ent.contentText);
          enhanceCodeBlocks(ent.content);
          ent.content.scrollTop = ent.content.scrollHeight;
          ent.contentCount.textContent = `(${ent.contentText.length} chars)`;
        });
      }
    }
    if (ev.reasoning) {
      ent.reasoningText += ev.reasoning;
      ent.reasoningSection.hidden = false;
      ent.reasoningSection.open = true;
      if (!ent._pendingReasoning) {
        ent._pendingReasoning = true;
        requestAnimationFrame(() => {
          ent._pendingReasoning = false;
          ent.reasoning.innerHTML = renderMarkdown(ent.reasoningText);
          enhanceCodeBlocks(ent.reasoning);
          ent.reasoning.scrollTop = ent.reasoning.scrollHeight;
          ent.reasoningCount.textContent = `(${ent.reasoningText.length} chars)`;
        });
      }
    }
    return;
  }
  if (ev.type === "request.end") {
    const ent = activity.byId.get(ev.id);
    activity.liveCount = Math.max(0, activity.liveCount - 1);
    if (!ev.ok) activity.errors++;
    refreshActivityCount();
    refreshActivityStats();
    if (!ent) return;
    ent.card.classList.remove("live");
    ent.card.dataset.status = ev.ok ? "ok" : "err";
    if (!ev.ok) ent.card.classList.add("err");
    const pill = ent.card.querySelector(".pill.live");
    if (pill) {
      pill.classList.remove("live");
      pill.classList.add(ev.ok ? "ok" : "err");
      pill.textContent = ev.ok ? `OK ${ev.status || ""}`.trim() : "ERR";
    }
    const dt = ev.duration_s ? `${ev.duration_s.toFixed(2)}s` : "?";
    const usage = ev.usage || {};
    const tok = usage.completion_tokens ?? usage.total_tokens;
    const tps = usage.predicted_per_second ?? null;
    if (typeof tps === "number" && isFinite(tps) && tps > 0) {
      activity.tpsSamples.push(tps);
      if (activity.tpsSamples.length > 50) activity.tpsSamples.shift();
      refreshActivityStats();
    }
    let line = `done in ${dt}`;
    if (typeof tok === "number") line += ` · ${tok} tok`;
    if (typeof tps === "number") line += ` · ${tps.toFixed(1)} tok/s`;
    if (typeof ev.chunks === "number") line += ` · ${ev.chunks} chunks`;
    if (ev.error) line += ` · error: ${ev.error}`;
    ent.card.querySelector(".status").textContent = line;
    applyActivityFilters();
    return;
  }
}

bindActivityUi();
// Auto-connect even if panel hidden so the count badge is always live.
connectActivity();

// ---------------- knowledge cards (RAG management) ----------------
let __kbDocsCache = [];

async function refreshKnowledgeCards() {
  const stats = $("#kb-stats");
  const grid = $("#kb-cards");
  const pills = $("#kb-source-pills");
  const src = $("#kb-filter-source").value.trim();
  const tag = $("#kb-filter-tag").value.trim();
  stats.textContent = "loading…";
  grid.innerHTML = "";
  if (pills) pills.innerHTML = "";
  try {
    const params = new URLSearchParams();
    if (src) params.set("source", src);
    if (tag) params.set("tag", tag);
    const [statsR, docsR] = await Promise.all([
      fetch("/rag/stats"),
      fetch(`/rag/documents?${params.toString()}`),
    ]);
    const s = statsR.ok ? await statsR.json() : { total_chunks: 0, sources: [] };
    const d = docsR.ok ? await docsR.json() : { documents: [] };
    const docs = d.documents || [];
    __kbDocsCache = docs;
    const srcCount = (s.sources || []).length;
    stats.innerHTML = `<strong>${s.total_chunks || 0}</strong> chunks · <strong>${docs.length}</strong> documents · <strong>${srcCount}</strong> sources`;
    if (pills) {
      pills.innerHTML = (s.sources || []).map(x =>
        `<span class="src-pill" data-src="${escapeAttr(x.source)}">${escapeHtml(x.source)} (${x.chunks})</span>`
      ).join("");
      pills.querySelectorAll(".src-pill").forEach(p => p.addEventListener("click", () => {
        $("#kb-filter-source").value = p.dataset.src;
        refreshKnowledgeCards();
      }));
    }
    if (!docs.length) {
      grid.innerHTML = `<div class="kb-empty">No knowledge cards${src || tag ? " matching the current filters" : ""} yet.</div>`;
    } else {
      for (const card of docs) renderKnowledgeCard(grid, card);
    }
    // If the network pane is currently visible, redraw with new data.
    const netPane = document.querySelector('.kb-pane[data-kbpane="network"]');
    if (netPane && netPane.classList.contains("active")) renderKnowledgeNetwork();
  } catch (e) {
    stats.textContent = `error: ${e.message || e}`;
  }
}

function renderKnowledgeCard(grid, card) {
  const div = document.createElement("div");
  div.className = "kb-card";
  const ts = card.ingested_at ? new Date(card.ingested_at * 1000).toLocaleString() : "";
  const tagsHtml = (card.tags || []).map(t => `<span class="kb-tag">${escapeHtml(t)}</span>`).join("");
  div.innerHTML = `
    <div class="kb-card-head">
      <div class="kb-card-title" title="${escapeAttr(card.doc_id || "")}"></div>
      <button class="kb-del ghost" type="button" title="Delete">✕</button>
    </div>
    <div class="kb-card-meta"><span class="kb-src"></span> · <span class="kb-chunks"></span> chunks${ts ? ` · ${ts}` : ""}</div>
    <div class="kb-card-tags">${tagsHtml}</div>
    <div class="kb-card-preview"></div>
  `;
  div.querySelector(".kb-card-title").textContent = card.title || card.doc_id;
  div.querySelector(".kb-src").textContent = card.source || "(none)";
  div.querySelector(".kb-chunks").textContent = card.chunks;
  const previewEl = div.querySelector(".kb-card-preview");
  // Render markdown so tables/headings/code don't appear as raw markup. We
  // strip ATX heading markers and the most aggressive whitespace so the
  // teaser stays compact at card size.
  const previewText = stripPreviewMd(card.preview || "");
  previewEl.classList.add("markdown");
  previewEl.innerHTML = renderMarkdown(previewText);
  enhanceCodeBlocks(previewEl);
  div.addEventListener("click", (e) => {
    if (e.target.closest(".kb-del")) return;
    openKbDrawer(card);
  });
  div.querySelector(".kb-del").addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!confirm(`Delete "${card.title || card.doc_id}" (${card.chunks} chunks)?`)) return;
    try {
      const r = await fetch("/rag/documents", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chunk_ids: card.chunk_ids }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      div.remove();
      refreshKnowledgeCards();
    } catch (err) {
      alert(`Delete failed: ${err.message || err}`);
    }
  });
  grid.appendChild(div);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function escapeAttr(s) { return escapeHtml(s); }

// ---------------- settings dialog ----------------
let __tavilyOriginal = "";
async function openSettingsDialog() {
  const s = state.settings || {};
  const setVal = (sel, v) => { const el = $(sel); if (el) el.value = (v ?? v === 0) ? v : ""; };
  setVal("#set-temperature", s.temperature);
  setVal("#set-top-p", s.top_p);
  setVal("#set-top-k", s.top_k);
  setVal("#set-max-tokens", s.max_tokens);
  setVal("#set-repeat-penalty", s.repeat_penalty);
  $("#set-system").value = s.system || "";
  // Load runtime config (server-side, redacted)
  const tavilyEl = $("#set-tavily-key");
  const status = $("#set-tavily-status");
  if (tavilyEl) {
    tavilyEl.value = "";
    __tavilyOriginal = "";
    if (status) status.textContent = "Loading…";
    try {
      const r = await fetch("/admin/runtime-config");
      if (r.ok) {
        const j = await r.json();
        const masked = (j && j.tavily_api_key) ? String(j.tavily_api_key) : "";
        if (masked) {
          tavilyEl.placeholder = masked + "  (set — leave blank to keep)";
          __tavilyOriginal = "__KEEP__";
          if (status) status.textContent = "Tavily key is set on the server.";
        } else {
          tavilyEl.placeholder = "tvly-…";
          if (status) status.textContent = "No Tavily key set — web_search will fall back to DuckDuckGo.";
        }
      } else if (status) {
        status.textContent = "Could not load runtime config (HTTP " + r.status + ").";
      }
    } catch (e) {
      if (status) status.textContent = "Could not contact server.";
    }
  }
  $("#settings-dialog").showModal();
}
async function readSettingsDialog() {
  const num = (sel) => {
    const v = $(sel).value.trim();
    if (v === "") return undefined;
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  };
  state.settings = {
    temperature: num("#set-temperature"),
    top_p: num("#set-top-p"),
    top_k: num("#set-top-k"),
    max_tokens: num("#set-max-tokens"),
    repeat_penalty: num("#set-repeat-penalty"),
    system: $("#set-system").value,
  };
  // Strip undefineds for cleanliness
  for (const k of Object.keys(state.settings)) if (state.settings[k] === undefined) delete state.settings[k];
  saveSettings();
  // Persist Tavily API key if changed
  const tavilyEl = $("#set-tavily-key");
  if (tavilyEl) {
    const v = tavilyEl.value;
    // Only POST if user typed something. Empty + had original means keep; empty + no original means no-op.
    // To clear, user types a single space; we treat trimmed empty as no-op.
    if (v && v.trim() && v.trim() !== "__KEEP__") {
      try {
        await fetch("/admin/runtime-config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ tavily_api_key: v.trim() }),
        });
      } catch {}
    }
    tavilyEl.value = "";
  }
}

// ---------------- attachments (multimodal) ----------------
async function onAttachImages(e) {
  const files = Array.from(e.target.files || []);
  e.target.value = "";
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    const url = await new Promise((res, rej) => {
      const fr = new FileReader();
      fr.onload = () => res(fr.result);
      fr.onerror = rej;
      fr.readAsDataURL(f);
    });
    PENDING_ATTACH.push(url);
  }
  renderAttachments();
}
function renderAttachments() {
  const box = $("#attachments");
  box.innerHTML = "";
  if (!PENDING_ATTACH.length && !PENDING_DOCS.length) { box.hidden = true; return; }
  box.hidden = false;
  PENDING_ATTACH.forEach((url, i) => {
    const wrap = document.createElement("div");
    wrap.className = "attach-thumb";
    const img = document.createElement("img");
    img.src = url;
    const x = document.createElement("button");
    x.type = "button";
    x.textContent = "✕";
    x.title = "Remove";
    x.addEventListener("click", () => { PENDING_ATTACH.splice(i, 1); renderAttachments(); });
    wrap.appendChild(img);
    wrap.appendChild(x);
    box.appendChild(wrap);
  });
  PENDING_DOCS.forEach((d, i) => {
    const wrap = document.createElement("div");
    wrap.className = "attach-doc";
    const ic = d.uploading ? "⏳" : (d.error ? "⚠" : "📄");
    const sz = d.size ? fmtBytes(d.size) : "";
    const tokens = d.text ? Math.ceil(d.text.length / 4) : null;
    const meta = [d.format || d.ext, sz, tokens ? `~${tokens} tok` : null].filter(Boolean).join(" · ");
    wrap.innerHTML = `
      <span class="attach-doc-ic">${ic}</span>
      <span class="attach-doc-name"></span>
      <span class="attach-doc-meta"></span>
      <button type="button" class="attach-doc-x" title="Remove">✕</button>
    `;
    wrap.querySelector(".attach-doc-name").textContent = d.name;
    wrap.querySelector(".attach-doc-meta").textContent = meta;
    if (d.error) wrap.classList.add("err");
    if (d.uploading) wrap.classList.add("loading");
    wrap.querySelector(".attach-doc-x").addEventListener("click", () => {
      PENDING_DOCS.splice(i, 1);
      renderAttachments();
    });
    box.appendChild(wrap);
  });
}

const _DOC_TEXT_EXTS = new Set([
  "txt","md","markdown","rst","log","csv","tsv","json","jsonl","ndjson",
  "yml","yaml","toml","ini","cfg","conf","html","htm","xml","svg",
  "py","js","ts","tsx","jsx","css","scss","c","h","cc","cpp","hpp",
  "rs","go","java","kt","rb","php","sh","bash","zsh","ps1","bat","sql",
]);

async function onAttachDocs(e) {
  const files = Array.from(e.target.files || []);
  e.target.value = "";
  for (const f of files) {
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    const placeholder = { name: f.name, ext, size: f.size, text: "", uploading: true };
    PENDING_DOCS.push(placeholder);
    renderAttachments();
    try {
      let text = "";
      let format = "text";
      let warning = null;
      if (_DOC_TEXT_EXTS.has(ext)) {
        text = await f.text();
        format = ext === "json" || ext === "jsonl" ? "json" : (ext === "csv" || ext === "tsv" ? "csv" : "text");
      } else {
        // Binary or unknown — extract on the server.
        const fd = new FormData();
        fd.append("file", f, f.name);
        const r = await fetch("/admin/extract", { method: "POST", body: fd });
        const j = await r.json();
        if (!r.ok) throw new Error(j.detail || `HTTP ${r.status}`);
        text = j.text || "";
        format = j.format || ext;
        warning = j.warning || null;
      }
      // Cap per-doc text to keep prompts sane.
      const cap = 200000;
      if (text.length > cap) {
        warning = (warning ? warning + "; " : "") + `truncated to ${cap} chars`;
        text = text.slice(0, cap) + "\n\n[…truncated]";
      }
      placeholder.text = text;
      placeholder.format = format;
      placeholder.warning = warning;
      placeholder.uploading = false;
    } catch (err) {
      placeholder.uploading = false;
      placeholder.error = err.message || String(err);
    }
    renderAttachments();
  }
}

// ---------------- platform detection / mobile drawer ----------------
function detectPlatform() {
  const ua = navigator.userAgent || "";
  const uaMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile Safari/i.test(ua);
  const coarse = matchMedia("(hover: none) and (pointer: coarse)").matches;
  const narrow = matchMedia("(max-width: 760px)").matches;
  const isMobile = (uaMobile || coarse) && narrow;
  const isTouch = uaMobile || coarse;
  return { isMobile, isTouch, narrow };
}

function applyPlatformClasses() {
  const { isMobile, isTouch } = detectPlatform();
  document.body.classList.toggle("is-mobile", isMobile);
  document.body.classList.toggle("is-touch", isTouch);
  document.body.classList.toggle("is-desktop", !isMobile);
}

function setupPlatform() {
  applyPlatformClasses();
  // React to viewport changes (orientation, browser resize, devtools).
  window.addEventListener("resize", applyPlatformClasses, { passive: true });
  window.addEventListener("orientationchange", applyPlatformClasses);

  // Sidebar drawer toggle (only matters on narrow viewports; on desktop the
  // sidebar is always visible and the toggle is hidden by CSS).
  const btn = document.getElementById("sidebar-toggle");
  const backdrop = document.getElementById("sidebar-backdrop");
  if (!btn || !backdrop) return;
  const close = () => {
    document.body.classList.remove("sidebar-open");
    backdrop.hidden = true;
  };
  const open = () => {
    document.body.classList.add("sidebar-open");
    backdrop.hidden = false;
  };
  btn.addEventListener("click", () => {
    if (document.body.classList.contains("sidebar-open")) close();
    else open();
  });
  backdrop.addEventListener("click", close);
  // Close drawer when picking a conversation or starting a new chat.
  document.getElementById("conversations")?.addEventListener("click", () => {
    if (matchMedia("(max-width: 760px)").matches) close();
  });
  document.getElementById("new-chat")?.addEventListener("click", () => {
    if (matchMedia("(max-width: 760px)").matches) close();
  });
  // Close drawer when changing tab on mobile, since the topbar is what's
  // visible while the drawer is closed.
  document.querySelectorAll(".topbar .tabs .tab").forEach((t) => {
    t.addEventListener("click", () => {
      if (matchMedia("(max-width: 760px)").matches) close();
    });
  });
  // Esc closes.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && document.body.classList.contains("sidebar-open")) close();
  });
}

// ---------------- composer drag & drop ----------------
function setupComposerDragDrop() {
  const zone = document.getElementById("composer-dropzone");
  const ta = document.getElementById("prompt");
  if (!zone || !ta) return;
  let depth = 0;
  const isFileDrag = (e) => {
    const dt = e.dataTransfer;
    if (!dt) return false;
    const types = Array.from(dt.types || []);
    return types.includes("Files");
  };
  zone.addEventListener("dragenter", (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    depth++;
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragover", (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  zone.addEventListener("dragleave", (e) => {
    if (!isFileDrag(e)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) zone.classList.remove("dragover");
  });
  zone.addEventListener("drop", async (e) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    depth = 0;
    zone.classList.remove("dragover");
    const files = Array.from(e.dataTransfer.files || []);
    if (!files.length) return;
    const imgs = files.filter((f) => (f.type || "").startsWith("image/"));
    const docs = files.filter((f) => !(f.type || "").startsWith("image/"));
    if (imgs.length) {
      await onAttachImages({ target: { files: imgs, value: "" } });
    }
    if (docs.length) {
      await onAttachDocs({ target: { files: docs, value: "" } });
    }
  });
  // Prevent the browser from navigating away if a stray drop misses the zone.
  window.addEventListener("dragover", (e) => { if (isFileDrag(e)) e.preventDefault(); });
  window.addEventListener("drop", (e) => { if (isFileDrag(e) && !zone.contains(e.target)) e.preventDefault(); });

  // ----- clipboard paste: images + files -----
  // Pasting an image (e.g. screenshot) or a file from the OS file manager
  // attaches it the same way drag & drop does. Plain text paste is left to
  // the textarea's default behaviour.
  const handlePaste = async (e) => {
    const dt = e.clipboardData;
    if (!dt) return;
    const files = [];
    if (dt.files && dt.files.length) {
      for (const f of dt.files) files.push(f);
    } else if (dt.items && dt.items.length) {
      for (const it of dt.items) {
        if (it.kind === "file") {
          const f = it.getAsFile();
          if (f) files.push(f);
        }
      }
    }
    if (!files.length) return; // let the browser handle plain-text paste
    e.preventDefault();
    const imgs = files.filter((f) => (f.type || "").startsWith("image/"));
    const docs = files.filter((f) => !(f.type || "").startsWith("image/"));
    if (imgs.length) await onAttachImages({ target: { files: imgs, value: "" } });
    if (docs.length) await onAttachDocs({ target: { files: docs, value: "" } });
  };
  ta.addEventListener("paste", handlePaste);
  // Also catch pastes anywhere in the document while the chat tab is active
  // and no other input/textarea is focused, so a screenshot from another app
  // can be pasted without first clicking the textarea.
  document.addEventListener("paste", (e) => {
    const ae = document.activeElement;
    const tag = (ae?.tagName || "").toLowerCase();
    const isOtherInput =
      ae && ae !== ta && (tag === "input" || tag === "textarea" || ae.isContentEditable);
    if (isOtherInput) return;
    const chatTabActive = document.querySelector('.tab-view[data-view="chat"].active');
    if (!chatTabActive) return;
    handlePaste(e);
  });
}

// ---------------- voice input (Whisper) ----------------
let _recState = null;
async function toggleVoiceRecord() {
  const btn = $("#mic-btn");
  if (_recState) {
    // Stop recording
    try { _recState.recorder.stop(); } catch { /* ignore */ }
    return;
  }
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    alert("Microphone or MediaRecorder not available in this browser.");
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    alert("Microphone permission denied: " + (e.message || e));
    return;
  }
  // Pick an OGG/WebM mime supported by the browser.
  const mimes = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/ogg", "audio/mp4"];
  const mime = mimes.find(m => MediaRecorder.isTypeSupported(m)) || "";
  const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  const chunks = [];
  recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  recorder.onstop = async () => {
    btn.classList.remove("recording");
    btn.textContent = "🎙";
    _recState = null;
    stream.getTracks().forEach(t => t.stop());
    if (!chunks.length) return;
    const blob = new Blob(chunks, { type: mime || "audio/webm" });
    const ext = (mime.split("/")[1] || "webm").split(";")[0];
    const fd = new FormData();
    fd.append("file", blob, `voice.${ext}`);
    fd.append("model", "base");
    btn.disabled = true;
    const prev = $("#composer-meta").textContent;
    $("#composer-meta").textContent = "transcribing…";
    try {
      const r = await fetch("/v1/audio/transcriptions", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || JSON.stringify(j));
      const ta = $("#prompt");
      const insert = (j.text || "").trim();
      if (insert) {
        ta.value = ta.value ? ta.value.replace(/\s+$/, "") + " " + insert : insert;
        ta.focus();
      }
      $("#composer-meta").textContent = prev || "";
    } catch (e) {
      $("#composer-meta").textContent = "transcribe failed: " + (e.message || e);
    } finally {
      btn.disabled = false;
    }
  };
  recorder.start();
  btn.classList.add("recording");
  btn.textContent = "⏺";
  _recState = { recorder, stream };
}

// ---------------- summarize from RAG dialog ----------------
async function summarizeAndIngest() {
  const status = $("#ing-status");
  const text = $("#ing-text").value.trim();
  if (!text) { status.textContent = "Paste text first."; return; }
  const title = $("#ing-title").value.trim() || "summary";
  const source = $("#ing-source").value.trim() || "summaries";
  const tags = $("#ing-tags").value.split(",").map(s => s.trim()).filter(Boolean);
  status.textContent = "summarizing via sub-agent…";
  try {
    const r = await fetch("/admin/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, title, source, tags: tags.length ? tags : ["summary"], unload_after: true }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || JSON.stringify(j));
    status.textContent = `summarized → ingested as “${j.title}” (${j.ingest?.chunks ?? 0} chunks)`;
    $("#ing-text").value = j.summary || "";
    refreshKnowledgeCards();
  } catch (e) {
    status.textContent = `error: ${e.message || e}`;
  }
}

// ---------------- tab switching ----------------
const __tabSeen = new Set();

function switchTab(name) {
  if (!name) return;
  document.querySelectorAll(".tabs .tab").forEach(b => {
    b.classList.toggle("active", b.dataset.tab === name);
  });
  document.querySelectorAll(".tab-view").forEach(v => {
    v.classList.toggle("active", v.dataset.view === name);
  });
  // Lazy-render heavy views the first time they become visible.
  if (!__tabSeen.has(name)) {
    __tabSeen.add(name);
    if (name === "models") renderModelsTab();
    else if (name === "knowledge") refreshKnowledgeCards();
    else if (name === "activity") { refreshActivityStats(); refreshActivityModelFilter(); }
  } else if (name === "models") {
    renderModelsTab();
  }
  if (name === "models") renderActiveStrip();
}

function switchKbView(name) {
  if (!name) return;
  document.querySelectorAll(".kb-view-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.kbview === name);
  });
  document.querySelectorAll(".kb-pane").forEach(p => {
    p.classList.toggle("active", p.dataset.kbpane === name);
  });
  if (name === "network") renderKnowledgeNetwork();
}

// ---------------- Models tab ----------------
async function renderModelsTab() {
  const grid = document.getElementById("models-grid");
  if (!grid) return;
  // Make sure admin status is fresh (path_exists / download specs).
  await refreshModels().catch(() => {});
  const health = await fetch("/health").then(r => r.ok ? r.json() : {}).catch(() => ({}));
  grid.innerHTML = "";
  const models = state.models || [];
  if (!models.length) {
    grid.innerHTML = `<div class="kb-empty">No models registered.</div>`;
    renderActiveStrip(health);
    return;
  }
  for (const m of models) {
    const admin = (state.modelAdmin || {})[m.id] || {};
    const card = document.createElement("div");
    card.className = "model-card";
    const isActive = (m.kind === "embedding" && health.embedder_ready)
      || (m.kind === "sub_agent" && health.sub_agent_ready)
      || (m.kind === "vision" && health.vision_ready)
      || (m.kind === "chat" && health.active_chat_model === m.id);
    if (isActive) card.classList.add("active");
    const exists = admin.path_exists !== false;
    const dl = state.downloads && state.downloads[m.id];
    const dlInProgress = dl && !["done", "error", "skip"].includes(dl.phase || "");
    const kindLabel = m.kind === "embedding" ? "embedding"
      : m.kind === "sub_agent" ? "sub-agent"
      : m.kind === "vision" ? "multi-modal"
      : "chat";
    card.innerHTML = `
      <div class="model-card-head">
        <div class="model-card-title"></div>
        <span class="model-kind ${m.kind}">${kindLabel}</span>
      </div>
      <div class="model-card-meta">
        <div class="row"><span class="k">folder:</span> <span class="folder"></span></div>
        <div class="row"><span class="k">path:</span> <span class="path"></span></div>
        <div class="row status">
          ${isActive ? '<span class="pill ok">loaded</span>' : '<span class="pill">idle</span>'}
          ${exists ? '<span class="pill">weights present</span>' : '<span class="pill warn">weights missing</span>'}
          ${admin.mmproj_exists === false && admin.mmproj ? '<span class="pill warn">mmproj missing</span>' : ""}
        </div>
      </div>
      <div class="model-card-progress" hidden>
        <div class="bar"><div class="fill"></div></div>
        <div class="ptxt"></div>
      </div>
      <div class="model-card-actions">
        <button type="button" class="btn-load" ${(m.kind === "embedding" || m.kind === "vision") ? "disabled" : ""}>${isActive ? "Reload" : "Load"}</button>
        <button type="button" class="btn-eject" ${isActive && m.kind === "chat" ? "" : "disabled"}>Eject</button>
        <button type="button" class="btn-fetch" ${admin.download && !exists && !dlInProgress ? "" : "hidden"}>⬇ Download</button>
        <button type="button" class="btn-args ghost">Args ▾</button>
      </div>
      <pre class="model-args" hidden></pre>
    `;
    card.querySelector(".model-card-title").textContent = m.id;
    card.querySelector(".folder").textContent = admin.folder || "(builtin)";
    card.querySelector(".path").textContent = admin.path || "(unset)";
    if (dlInProgress) {
      const wrap = card.querySelector(".model-card-progress");
      wrap.hidden = false;
      const pct = (dl.total && dl.downloaded) ? Math.round((dl.downloaded / dl.total) * 100) : null;
      wrap.querySelector(".fill").style.width = pct != null ? `${pct}%` : "20%";
      wrap.querySelector(".ptxt").textContent = `${dl.phase || "downloading"}${pct != null ? ` · ${pct}%` : ""}`;
    }
    // Args expandable.
    card.querySelector(".btn-args").addEventListener("click", () => {
      const pre = card.querySelector(".model-args");
      if (pre.hidden) {
        pre.textContent = JSON.stringify(admin.args || m.args || {}, null, 2);
        pre.hidden = false;
      } else {
        pre.hidden = true;
      }
    });
    // Load: send a tiny chat ping (chat) or warm via /admin/warmup if available.
    card.querySelector(".btn-load").addEventListener("click", async () => {
      const btn = card.querySelector(".btn-load");
      btn.disabled = true;
      btn.textContent = "Loading…";
      try {
        if (m.kind === "chat") {
          const r = await fetch("/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: m.id, messages: [{ role: "user", content: "ping" }], max_tokens: 1, stream: false }),
          });
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
        } else if (m.kind === "sub_agent") {
          const r = await fetch("/admin/summarize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: "warmup", title: "warmup", source: "_warmup", tags: [], unload_after: false }),
          });
          // 4xx allowed; we just want the model loaded
          if (r.status >= 500) throw new Error(`HTTP ${r.status}`);
        }
      } catch (e) {
        alert(`Load failed: ${e.message || e}`);
      } finally {
        renderModelsTab();
        refreshHealth();
      }
    });
    card.querySelector(".btn-eject").addEventListener("click", async () => {
      try {
        const r = await fetch("/admin/unload", { method: "POST" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
      } catch (e) {
        alert(`Eject failed: ${e.message || e}`);
      } finally {
        renderModelsTab();
        refreshHealth();
      }
    });
    const fetchBtn = card.querySelector(".btn-fetch");
    if (fetchBtn) fetchBtn.addEventListener("click", async () => {
      fetchBtn.disabled = true;
      fetchBtn.textContent = "Queued…";
      try {
        const r = await fetch("/admin/fetch-model", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: m.id }),
        });
        if (!r.ok) {
          const j = await r.json().catch(() => ({}));
          throw new Error(j.detail || `HTTP ${r.status}`);
        }
      } catch (e) {
        alert(`Download failed: ${e.message || e}`);
        fetchBtn.disabled = false;
        fetchBtn.textContent = "⬇ Download";
      }
    });
    grid.appendChild(card);
  }
  renderActiveStrip(health);
}

async function renderActiveStrip(health) {
  const strip = document.getElementById("active-strip");
  if (!strip) return;
  if (!health) {
    try { health = await fetch("/health").then(r => r.json()); } catch { health = {}; }
  }
  const pills = [];
  if (health.active_chat_model) pills.push(`<span class="active-pill chat">chat: <strong>${escapeHtml(health.active_chat_model)}</strong></span>`);
  else pills.push(`<span class="active-pill idle">no chat model loaded</span>`);
  if (health.sub_agent_ready) pills.push(`<span class="active-pill sub">sub-agent ready</span>`);
  if (health.vision_ready) pills.push(`<span class="active-pill vision">multi-modal ready</span>`);
  if (health.embedder_ready) pills.push(`<span class="active-pill emb">embedder ready</span>`);
  strip.innerHTML = pills.join("");
}

// ---------------- Knowledge network (interactive 2D + 3D) ----------------
//
// Shared state across renders so the 2D pan/zoom transform survives a
// re-layout, and the 3D scene can be lazy-initialised on first toggle.
const __kbNet = {
  view: "2d",            // "2d" | "3d"
  mode: "both",          // link mode (tag / source / both)
  nodes: [],
  edges: [],
  // 2D pan/zoom transform applied to the <g class="viewport"> group.
  tx: 0, ty: 0, scale: 1,
  // Selected/hovered node id (kept across re-renders).
  selectedId: null,
  // 3D state, populated on first switch to 3D mode.
  three: null,
};

function _kbNetRoot() { return document.getElementById("kb-network"); }
function _kbNetSvg()  { return document.getElementById("kb-network-svg"); }
function _kbNet3d()   { return document.getElementById("kb-network-3d"); }
function _kbNetCard() { return document.getElementById("kb-node-card"); }

function _markNetInteracted() {
  const root = _kbNetRoot();
  if (root) root.classList.add("interacted");
}

function _kbBuildGraph(docs, mode) {
  const nodes = docs.map((d) => ({
    id: `${d.source}::${d.doc_id}`,
    label: d.title || d.doc_id,
    source: d.source || "",
    tags: d.tags || [],
    chunks: d.chunks || 1,
    doc: d,
  }));
  const edges = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      let link = 0;
      const shared = (mode === "tag" || mode === "both")
        ? a.tags.filter(t => b.tags.includes(t)) : [];
      link += shared.length;
      if ((mode === "source" || mode === "both") && a.source && a.source === b.source) link += 1;
      if (link > 0) edges.push({ a: i, b: j, w: link, shared });
    }
  }
  return { nodes, edges };
}

function _kbColorFor(idx) {
  const palette = ["#4f8cff", "#6dd58c", "#f3a85c", "#cf6dff", "#ff6d8a", "#5cd5cf", "#e3d35c", "#ff9f6d"];
  return palette[((idx % palette.length) + palette.length) % palette.length];
}

// Lay out nodes in 2D with simple force-directed iterations.
function _kbLayout2D(nodes, edges, W, H) {
  for (const n of nodes) {
    n.x = W / 2 + (Math.random() - 0.5) * W * 0.6;
    n.y = H / 2 + (Math.random() - 0.5) * H * 0.6;
    n.vx = 0; n.vy = 0;
  }
  const iters = 140;
  const k = Math.sqrt((W * H) / Math.max(nodes.length, 1)) * 0.6;
  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
        const f = (k * k) / dist;
        dx /= dist; dy /= dist;
        a.vx += dx * f * 0.05; a.vy += dy * f * 0.05;
        b.vx -= dx * f * 0.05; b.vy -= dy * f * 0.05;
      }
    }
    for (const e of edges) {
      const a = nodes[e.a], b = nodes[e.b];
      let dx = a.x - b.x, dy = a.y - b.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = (dist * dist) / k * Math.min(e.w, 4) * 0.02;
      dx /= dist; dy /= dist;
      a.vx -= dx * f; a.vy -= dy * f;
      b.vx += dx * f; b.vy += dy * f;
    }
    const cx = W / 2, cy = H / 2;
    for (const n of nodes) {
      n.vx += (cx - n.x) * 0.005;
      n.vy += (cy - n.y) * 0.005;
      n.vx *= 0.85; n.vy *= 0.85;
      const v = Math.sqrt(n.vx * n.vx + n.vy * n.vy);
      if (v > 20) { n.vx = n.vx / v * 20; n.vy = n.vy / v * 20; }
      n.x += n.vx; n.y += n.vy;
      n.x = Math.max(20, Math.min(W - 20, n.x));
      n.y = Math.max(20, Math.min(H - 20, n.y));
    }
  }
}

function renderKnowledgeNetwork() {
  const root = _kbNetRoot();
  if (!root) return;
  const docs = __kbDocsCache || [];
  const modeEl = document.getElementById("kb-net-mode");
  __kbNet.mode = modeEl ? modeEl.value : "both";

  const { nodes, edges } = _kbBuildGraph(docs, __kbNet.mode);
  __kbNet.nodes = nodes;
  __kbNet.edges = edges;

  // Stats line.
  const stats = document.getElementById("kb-net-stats");
  if (stats) stats.textContent = nodes.length
    ? `${nodes.length} doc${nodes.length === 1 ? "" : "s"} · ${edges.length} link${edges.length === 1 ? "" : "s"}`
    : "no documents";

  if (__kbNet.view === "3d") {
    _renderKnowledgeNetwork3D();
  } else {
    _renderKnowledgeNetwork2D();
  }
  _hideNodeCard();
}

function _renderKnowledgeNetwork2D() {
  const svg = _kbNetSvg();
  const root = _kbNetRoot();
  if (!svg || !root) return;
  // Make sure the SVG layer is visible and the 3D one is hidden.
  svg.hidden = false;
  svg.style.display = "block";
  const c3d = _kbNet3d();
  if (c3d) { c3d.hidden = true; c3d.style.display = "none"; }

  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const { nodes, edges } = __kbNet;
  if (!nodes.length) {
    const NS = "http://www.w3.org/2000/svg";
    const txt = document.createElementNS(NS, "text");
    txt.setAttribute("x", "50%"); txt.setAttribute("y", "50%");
    txt.setAttribute("text-anchor", "middle");
    txt.setAttribute("class", "kb-net-empty");
    txt.textContent = "No knowledge yet.";
    svg.appendChild(txt);
    return;
  }
  const rect = svg.getBoundingClientRect();
  const W = rect.width || 800, H = rect.height || 500;
  _kbLayout2D(nodes, edges, W, H);

  const sources = Array.from(new Set(nodes.map(n => n.source || "(none)")));

  const NS = "http://www.w3.org/2000/svg";
  // Viewport group — pan/zoom transform applied here.
  const vp = document.createElementNS(NS, "g");
  vp.setAttribute("class", "viewport");
  svg.appendChild(vp);
  // Apply current transform.
  const applyTransform = () => {
    vp.setAttribute("transform",
      `translate(${__kbNet.tx},${__kbNet.ty}) scale(${__kbNet.scale})`);
  };
  applyTransform();

  // Edges first so nodes draw on top.
  const edgeEls = [];
  for (const e of edges) {
    const a = nodes[e.a], b = nodes[e.b];
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
    line.setAttribute("class", "edge");
    line.setAttribute("stroke-width", String(Math.min(1 + e.w * 0.4, 3)));
    line.dataset.a = e.a; line.dataset.b = e.b;
    vp.appendChild(line);
    edgeEls.push(line);
  }
  // Nodes.
  const nodeEls = [];
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "node");
    g.setAttribute("transform", `translate(${n.x},${n.y})`);
    g.dataset.idx = String(i);
    const r = Math.min(6 + Math.sqrt(n.chunks) * 2, 18);
    const circ = document.createElementNS(NS, "circle");
    circ.setAttribute("r", String(r));
    const sIdx = sources.indexOf(n.source || "(none)");
    circ.setAttribute("fill", _kbColorFor(sIdx));
    g.appendChild(circ);
    const label = document.createElementNS(NS, "text");
    label.setAttribute("x", "0");
    label.setAttribute("y", String(r + 12));
    label.textContent = (n.label || "").slice(0, 28);
    g.appendChild(label);
    g.addEventListener("mouseenter", () => _highlightNode(i));
    g.addEventListener("mouseleave", () => _highlightNode(null));
    g.addEventListener("click", (ev) => {
      ev.stopPropagation();
      _selectNode(i, true);
    });
    vp.appendChild(g);
    nodeEls.push(g);
  }
  // Legend (rendered outside the viewport so it stays put while panning/zooming).
  const legend = document.createElementNS(NS, "g");
  legend.setAttribute("class", "legend");
  legend.setAttribute("transform", "translate(10, 10)");
  const lh = Math.min(sources.length, 8) * 16 + 8;
  const lbg = document.createElementNS(NS, "rect");
  lbg.setAttribute("class", "bg");
  lbg.setAttribute("x", "0"); lbg.setAttribute("y", "0");
  lbg.setAttribute("width", "150"); lbg.setAttribute("height", String(lh));
  legend.appendChild(lbg);
  sources.slice(0, 8).forEach((s, i) => {
    const row = document.createElementNS(NS, "g");
    row.setAttribute("transform", `translate(8, ${8 + i * 16})`);
    const sw = document.createElementNS(NS, "rect");
    sw.setAttribute("width", "10"); sw.setAttribute("height", "10");
    sw.setAttribute("rx", "2"); sw.setAttribute("ry", "2");
    sw.setAttribute("fill", _kbColorFor(i));
    row.appendChild(sw);
    const tx = document.createElementNS(NS, "text");
    tx.setAttribute("x", "16"); tx.setAttribute("y", "10");
    tx.textContent = s.length > 20 ? s.slice(0, 19) + "…" : s;
    row.appendChild(tx);
    legend.appendChild(row);
  });
  svg.appendChild(legend);

  // Save renderer references for highlight + card positioning.
  __kbNet.dom = { svg, vp, edgeEls, nodeEls, applyTransform };

  // Restore selection if still present.
  if (__kbNet.selectedId) {
    const idx = nodes.findIndex(n => n.id === __kbNet.selectedId);
    if (idx >= 0) _selectNode(idx, false);
  }

  _setupNetInteractions();
}

// Wire pan/zoom + outside-click to dismiss card. Idempotent.
function _setupNetInteractions() {
  const root = _kbNetRoot();
  if (!root || root._kbWired) return;
  root._kbWired = true;

  const apply = () => __kbNet.dom?.applyTransform?.();

  // Mouse / touch pan.
  let dragging = false, lastX = 0, lastY = 0, moved = false;
  let pinch = null; // {dist, midX, midY, startScale}

  const pointerDown = (e) => {
    if (__kbNet.view === "3d") return; // 3D handles its own gestures
    if (e.target.closest(".node")) return; // node clicks pass through
    if (e.target.closest(".kb-node-card")) return;
    dragging = true; moved = false;
    lastX = e.clientX; lastY = e.clientY;
    root.classList.add("dragging");
    _markNetInteracted();
    if (e.pointerId !== undefined && root.setPointerCapture) {
      try { root.setPointerCapture(e.pointerId); } catch {}
    }
  };
  const pointerMove = (e) => {
    if (__kbNet.view === "3d") return;
    if (!dragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    lastX = e.clientX; lastY = e.clientY;
    __kbNet.tx += dx; __kbNet.ty += dy;
    apply();
    if (__kbNet.selectedId) _positionCardFromSelected();
  };
  const pointerUp = (e) => {
    if (__kbNet.view === "3d") return;
    if (!dragging) return;
    dragging = false;
    root.classList.remove("dragging");
    if (!moved) _hideNodeCard();
  };
  root.addEventListener("pointerdown", pointerDown);
  root.addEventListener("pointermove", pointerMove);
  root.addEventListener("pointerup", pointerUp);
  root.addEventListener("pointercancel", pointerUp);
  root.addEventListener("pointerleave", pointerUp);

  // Wheel zoom (cursor as zoom origin).
  root.addEventListener("wheel", (e) => {
    if (__kbNet.view === "3d") return;
    e.preventDefault();
    _markNetInteracted();
    const rect = root.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.0015);
    _zoomAt(cx, cy, factor);
  }, { passive: false });

  // Touch pinch (using two-finger pointer events isn't fully cross-browser,
  // so add a touch-event fallback).
  root.addEventListener("touchstart", (e) => {
    if (__kbNet.view === "3d") return;
    if (e.touches.length === 2) {
      const [a, b] = e.touches;
      const dx = a.clientX - b.clientX, dy = a.clientY - b.clientY;
      pinch = {
        dist: Math.hypot(dx, dy),
        midX: (a.clientX + b.clientX) / 2,
        midY: (a.clientY + b.clientY) / 2,
        startScale: __kbNet.scale,
      };
    }
  }, { passive: true });
  root.addEventListener("touchmove", (e) => {
    if (__kbNet.view === "3d" || !pinch || e.touches.length !== 2) return;
    e.preventDefault();
    const [a, b] = e.touches;
    const dx = a.clientX - b.clientX, dy = a.clientY - b.clientY;
    const dist = Math.hypot(dx, dy) || 1;
    const factor = (dist / pinch.dist);
    const rect = root.getBoundingClientRect();
    const cx = pinch.midX - rect.left, cy = pinch.midY - rect.top;
    // Re-apply factor relative to scale at pinch start.
    const target = Math.max(0.2, Math.min(6, pinch.startScale * factor));
    const cur = __kbNet.scale;
    _zoomAt(cx, cy, target / cur);
  }, { passive: false });
  root.addEventListener("touchend", () => { pinch = null; }, { passive: true });

  // Toolbar buttons.
  document.getElementById("kb-net-zoom-in")?.addEventListener("click", () => {
    const r = root.getBoundingClientRect();
    _zoomAt(r.width / 2, r.height / 2, 1.25);
  });
  document.getElementById("kb-net-zoom-out")?.addEventListener("click", () => {
    const r = root.getBoundingClientRect();
    _zoomAt(r.width / 2, r.height / 2, 1 / 1.25);
  });
  document.getElementById("kb-net-fit")?.addEventListener("click", () => {
    if (__kbNet.view === "3d") {
      _kb3dResetCamera();
    } else {
      __kbNet.tx = 0; __kbNet.ty = 0; __kbNet.scale = 1;
      apply();
      if (__kbNet.selectedId) _positionCardFromSelected();
    }
  });

  // 2D / 3D mode toggle.
  document.querySelectorAll(".kb-net-mode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const v = btn.dataset.kbnetview;
      if (v === __kbNet.view) return;
      document.querySelectorAll(".kb-net-mode-btn").forEach(b => b.classList.toggle("active", b === btn));
      __kbNet.view = v;
      renderKnowledgeNetwork();
    });
  });

  // Outside click on the canvas dismisses the card.
  root.addEventListener("click", (e) => {
    if (e.target.closest(".node") || e.target.closest(".kb-node-card")) return;
    _hideNodeCard();
  });
}

function _zoomAt(cx, cy, factor) {
  const newScale = Math.max(0.2, Math.min(6, __kbNet.scale * factor));
  const ratio = newScale / __kbNet.scale;
  __kbNet.tx = cx - (cx - __kbNet.tx) * ratio;
  __kbNet.ty = cy - (cy - __kbNet.ty) * ratio;
  __kbNet.scale = newScale;
  __kbNet.dom?.applyTransform?.();
  if (__kbNet.selectedId) _positionCardFromSelected();
}

function _highlightNode(idx) {
  if (__kbNet.view === "3d") {
    __kbNet.three?.highlight?.(idx);
    return;
  }
  const dom = __kbNet.dom;
  if (!dom) return;
  if (idx == null) {
    dom.nodeEls.forEach(g => g.classList.remove("hl", "dim"));
    dom.edgeEls.forEach(l => l.classList.remove("hl", "dim"));
    return;
  }
  const conn = new Set([idx]);
  __kbNet.edges.forEach(e => {
    if (e.a === idx) conn.add(e.b);
    if (e.b === idx) conn.add(e.a);
  });
  dom.nodeEls.forEach((g, i) => {
    g.classList.toggle("hl", i === idx);
    g.classList.toggle("dim", !conn.has(i));
  });
  dom.edgeEls.forEach((l) => {
    const a = +l.dataset.a, b = +l.dataset.b;
    const touch = a === idx || b === idx;
    l.classList.toggle("hl", touch);
    l.classList.toggle("dim", !touch);
  });
}

function _selectNode(idx, openDrawerAfter) {
  if (idx == null || idx < 0 || !__kbNet.nodes[idx]) return;
  const n = __kbNet.nodes[idx];
  __kbNet.selectedId = n.id;
  _highlightNode(idx);
  _showNodeCard(n, openDrawerAfter);
}

function _showNodeCard(node, allowOpen) {
  const card = _kbNetCard();
  if (!card) return;
  const tags = (node.tags || []).slice(0, 6).map(t =>
    `<span class="tag">${escapeHtml(t)}</span>`).join("");
  card.innerHTML = `
    <h4 class="knc-title"></h4>
    <div class="knc-meta">
      <span><strong>Source:</strong> </span>
      <span><strong>Chunks:</strong> ${node.chunks}</span>
    </div>
    <div class="knc-tags">${tags}</div>
    <div class="knc-actions">
      <button type="button" class="primary" data-knc="open">Open</button>
      <button type="button" class="ghost" data-knc="close">Close</button>
    </div>
  `;
  card.querySelector(".knc-title").textContent = node.label || node.id;
  card.querySelectorAll(".knc-meta span")[0].innerHTML =
    `<strong>Source:</strong> ${escapeHtml(node.source || "(none)")}`;
  card.hidden = false;
  // Trigger transition after layout.
  requestAnimationFrame(() => card.classList.add("shown"));
  card.querySelector('[data-knc="open"]').addEventListener("click", () => {
    openKbDrawer(node.doc);
  });
  card.querySelector('[data-knc="close"]').addEventListener("click", _hideNodeCard);
  _positionCardFromSelected();
}

function _hideNodeCard() {
  const card = _kbNetCard();
  if (card) {
    card.classList.remove("shown");
    card.hidden = true;
  }
  __kbNet.selectedId = null;
  _highlightNode(null);
}

function _positionCardFromSelected() {
  const card = _kbNetCard();
  if (!card || card.hidden) return;
  const id = __kbNet.selectedId;
  if (!id) return;
  const idx = __kbNet.nodes.findIndex(n => n.id === id);
  if (idx < 0) return;
  const n = __kbNet.nodes[idx];
  let x, y;
  if (__kbNet.view === "3d" && __kbNet.three) {
    const screen = _kb3dProject(n);
    if (!screen) return;
    x = screen.x; y = screen.y;
  } else {
    // SVG point in viewport-transformed space.
    x = __kbNet.tx + n.x * __kbNet.scale;
    y = __kbNet.ty + n.y * __kbNet.scale;
  }
  card.style.left = `${x}px`;
  card.style.top = `${y}px`;
}

// ---------- 3D mode ----------
function _kbMakeLabelSprite(THREE, text) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const fontPx = 28;
  const padX = 14, padY = 8;
  const measure = document.createElement("canvas").getContext("2d");
  measure.font = `600 ${fontPx}px system-ui, sans-serif`;
  const w = Math.ceil(measure.measureText(text).width) + padX * 2;
  const h = fontPx + padY * 2;
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil(w * dpr);
  canvas.height = Math.ceil(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.fillStyle = "rgba(15,18,24,0.78)";
  const r = 8;
  ctx.beginPath();
  ctx.moveTo(r, 0); ctx.lineTo(w - r, 0); ctx.quadraticCurveTo(w, 0, w, r);
  ctx.lineTo(w, h - r); ctx.quadraticCurveTo(w, h, w - r, h);
  ctx.lineTo(r, h); ctx.quadraticCurveTo(0, h, 0, h - r);
  ctx.lineTo(0, r); ctx.quadraticCurveTo(0, 0, r, 0);
  ctx.closePath(); ctx.fill();
  ctx.fillStyle = "#f4f6fb";
  ctx.font = `600 ${fontPx}px system-ui, sans-serif`;
  ctx.textBaseline = "middle";
  ctx.fillText(text, padX, h / 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  // Scale so the sprite has roughly the same on-screen footprint as the label aspect.
  const baseScale = 0.45;
  sprite.scale.set(w * baseScale, h * baseScale, 1);
  return sprite;
}

async function _ensureThree() {
  if (window.THREE && window.THREE._withOrbit) return window.THREE;
  if (!window.THREE) {
    await new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js";
      s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }
  // We don't depend on OrbitControls — implement minimal mouse rotate/zoom inline.
  window.THREE._withOrbit = true;
  return window.THREE;
}

function _kbLayout3D(nodes, edges) {
  // Spherical seed, then a few force-directed iterations in 3D.
  const R = 220;
  for (let i = 0; i < nodes.length; i++) {
    const t = i / Math.max(nodes.length, 1);
    const phi = Math.acos(1 - 2 * t);
    const theta = Math.PI * (1 + Math.sqrt(5)) * i;
    nodes[i].x3 = R * Math.cos(theta) * Math.sin(phi);
    nodes[i].y3 = R * Math.sin(theta) * Math.sin(phi);
    nodes[i].z3 = R * Math.cos(phi);
    nodes[i].vx3 = nodes[i].vy3 = nodes[i].vz3 = 0;
  }
  const k = 80;
  for (let it = 0; it < 80; it++) {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        let dx = a.x3 - b.x3, dy = a.y3 - b.y3, dz = a.z3 - b.z3;
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.01;
        const f = (k * k) / d;
        dx /= d; dy /= d; dz /= d;
        a.vx3 += dx * f * 0.04; a.vy3 += dy * f * 0.04; a.vz3 += dz * f * 0.04;
        b.vx3 -= dx * f * 0.04; b.vy3 -= dy * f * 0.04; b.vz3 -= dz * f * 0.04;
      }
    }
    for (const e of edges) {
      const a = nodes[e.a], b = nodes[e.b];
      let dx = a.x3 - b.x3, dy = a.y3 - b.y3, dz = a.z3 - b.z3;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.01;
      const f = (d * d) / k * Math.min(e.w, 4) * 0.015;
      dx /= d; dy /= d; dz /= d;
      a.vx3 -= dx * f; a.vy3 -= dy * f; a.vz3 -= dz * f;
      b.vx3 += dx * f; b.vy3 += dy * f; b.vz3 += dz * f;
    }
    for (const n of nodes) {
      // Mild gravity to origin.
      n.vx3 += -n.x3 * 0.002;
      n.vy3 += -n.y3 * 0.002;
      n.vz3 += -n.z3 * 0.002;
      n.vx3 *= 0.85; n.vy3 *= 0.85; n.vz3 *= 0.85;
      n.x3 += n.vx3; n.y3 += n.vy3; n.z3 += n.vz3;
    }
  }
}

async function _renderKnowledgeNetwork3D() {
  const host = _kbNet3d();
  const svg = _kbNetSvg();
  if (!host) return;
  if (svg) { svg.hidden = true; svg.style.display = "none"; }
  host.hidden = false;
  host.style.display = "block";

  const THREE = await _ensureThree();
  const { nodes, edges } = __kbNet;
  _kbLayout3D(nodes, edges);

  // Rebuild scene from scratch so re-renders pick up new docs.
  if (__kbNet.three) {
    __kbNet.three.dispose();
    __kbNet.three = null;
  }

  const rect = host.getBoundingClientRect();
  const W = rect.width || 800, H = rect.height || 500;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, W / H, 1, 4000);
  camera.position.set(0, 0, 600);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(W, H, false);
  renderer.setClearColor(0x000000, 0);
  host.innerHTML = "";
  host.appendChild(renderer.domElement);
  renderer.domElement.style.cursor = "grab";

  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const dir = new THREE.DirectionalLight(0xffffff, 0.6);
  dir.position.set(200, 300, 400);
  scene.add(dir);

  // Edge mesh: one big LineSegments so it's cheap.
  const linePositions = new Float32Array(edges.length * 6);
  for (let i = 0; i < edges.length; i++) {
    const a = nodes[edges[i].a], b = nodes[edges[i].b];
    linePositions.set([a.x3, a.y3, a.z3, b.x3, b.y3, b.z3], i * 6);
  }
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
  const lineMat = new THREE.LineBasicMaterial({ color: 0x4f8cff, transparent: true, opacity: 0.35 });
  const lineSeg = new THREE.LineSegments(lineGeo, lineMat);
  scene.add(lineSeg);

  // Node spheres.
  const sources = Array.from(new Set(nodes.map(n => n.source || "(none)")));
  const meshes = [];
  const labelSprites = [];
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    const r = Math.min(6 + Math.sqrt(n.chunks) * 2, 18);
    const geo = new THREE.SphereGeometry(r, 24, 16);
    const sIdx = sources.indexOf(n.source || "(none)");
    const colorHex = parseInt(_kbColorFor(sIdx).slice(1), 16);
    const mat = new THREE.MeshLambertMaterial({ color: colorHex });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(n.x3, n.y3, n.z3);
    mesh.userData.idx = i;
    scene.add(mesh);
    meshes.push(mesh);
    // Title sprite floating above the sphere.
    const sprite = _kbMakeLabelSprite(THREE, (n.label || "").slice(0, 28));
    sprite.position.set(n.x3, n.y3 + r + 10, n.z3);
    sprite.userData.idx = i;
    sprite.userData.offset = r + 10;
    scene.add(sprite);
    labelSprites.push(sprite);
  }

  // Camera state — yaw/pitch around origin at distance.
  const cam = { yaw: 0, pitch: 0, dist: 600, target: new THREE.Vector3() };
  const updateCamera = () => {
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    camera.position.set(
      cam.target.x + cam.dist * cp * sy,
      cam.target.y + cam.dist * sp,
      cam.target.z + cam.dist * cp * cy,
    );
    camera.lookAt(cam.target);
  };
  updateCamera();

  // Pointer interactions: drag = rotate, wheel = dolly, click = pick.
  const root = _kbNetRoot();
  let dragging = false, lastX = 0, lastY = 0, moved = false;
  let pinch = null;
  const onDown = (e) => {
    if (e.target !== renderer.domElement) return;
    dragging = true; moved = false;
    lastX = e.clientX; lastY = e.clientY;
    root.classList.add("dragging");
    _markNetInteracted();
    if (e.pointerId !== undefined && renderer.domElement.setPointerCapture) {
      try { renderer.domElement.setPointerCapture(e.pointerId); } catch {}
    }
  };
  const onMove = (e) => {
    if (dragging) {
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
      lastX = e.clientX; lastY = e.clientY;
      cam.yaw   -= dx * 0.005;
      cam.pitch -= dy * 0.005;
      cam.pitch = Math.max(-1.4, Math.min(1.4, cam.pitch));
      updateCamera();
      if (__kbNet.selectedId) _positionCardFromSelected();
      return;
    }
    // Not dragging — hover-pick to show the floating knowledge card.
    const idx = pickAt(e.clientX, e.clientY);
    if (idx == null) {
      if (__kbNet.selectedId && !__kbNet.cardLocked) _hideNodeCard();
      renderer.domElement.style.cursor = "grab";
      return;
    }
    renderer.domElement.style.cursor = "pointer";
    const n = __kbNet.nodes[idx];
    if (!n) return;
    if (__kbNet.selectedId !== n.id) _selectNode(idx, false);
  };
  const onUp = (e) => {
    if (!dragging) return;
    dragging = false;
    root.classList.remove("dragging");
    if (!moved) {
      const idx = pickAt(e.clientX, e.clientY);
      if (idx != null) {
        // Defer the drawer open slightly so a dblclick (zoom) can pre-empt it.
        const n = __kbNet.nodes[idx];
        if (n) {
          if (__kbNet._clickTimer) clearTimeout(__kbNet._clickTimer);
          __kbNet._clickTimer = setTimeout(() => {
            __kbNet._clickTimer = null;
            _hideNodeCard();
            openKbDrawer(n.doc);
          }, 240);
        }
      } else {
        _hideNodeCard();
      }
    }
  };
  const onLeave = () => {
    if (!dragging && __kbNet.selectedId) _hideNodeCard();
  };
  const onWheel = (e) => {
    e.preventDefault();
    _markNetInteracted();
    cam.dist *= Math.exp(e.deltaY * 0.001);
    cam.dist = Math.max(120, Math.min(2000, cam.dist));
    updateCamera();
    if (__kbNet.selectedId) _positionCardFromSelected();
  };
  const onTouchStart = (e) => {
    if (e.touches.length === 2) {
      const [a, b] = e.touches;
      pinch = { dist: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY), startDist: cam.dist };
    }
  };
  const onTouchMove = (e) => {
    if (pinch && e.touches.length === 2) {
      e.preventDefault();
      const [a, b] = e.touches;
      const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY) || 1;
      cam.dist = Math.max(120, Math.min(2000, pinch.startDist * (pinch.dist / d)));
      updateCamera();
    }
  };
  const onTouchEnd = () => { pinch = null; };

  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const pickAt = (clientX, clientY) => {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.x = ((clientX - r.left) / r.width) * 2 - 1;
    ndc.y = -((clientY - r.top) / r.height) * 2 + 1;
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObjects(meshes, false);
    return hits.length ? hits[0].object.userData.idx : null;
  };

  renderer.domElement.addEventListener("pointerdown", onDown);
  renderer.domElement.addEventListener("pointermove", onMove);
  renderer.domElement.addEventListener("pointerup", onUp);
  renderer.domElement.addEventListener("pointercancel", onUp);
  renderer.domElement.addEventListener("pointerleave", onLeave);
  renderer.domElement.addEventListener("dblclick", (e) => {
    const idx = pickAt(e.clientX, e.clientY);
    if (idx == null) return;
    e.preventDefault();
    if (__kbNet._clickTimer) { clearTimeout(__kbNet._clickTimer); __kbNet._clickTimer = null; }
    _kb3dZoomToNeighborhood(idx);
  });
  renderer.domElement.addEventListener("wheel", onWheel, { passive: false });
  renderer.domElement.addEventListener("touchstart", onTouchStart, { passive: true });
  renderer.domElement.addEventListener("touchmove", onTouchMove, { passive: false });
  renderer.domElement.addEventListener("touchend", onTouchEnd);

  // Resize handling.
  const onResize = () => {
    const r2 = host.getBoundingClientRect();
    if (r2.width && r2.height) {
      renderer.setSize(r2.width, r2.height, false);
      camera.aspect = r2.width / r2.height;
      camera.updateProjectionMatrix();
    }
  };
  const ro = new ResizeObserver(onResize);
  ro.observe(host);

  // Render loop.
  let raf = 0;
  const tick = () => {
    renderer.render(scene, camera);
    if (__kbNet.selectedId) _positionCardFromSelected();
    raf = requestAnimationFrame(tick);
  };
  tick();

  __kbNet.three = {
    THREE, scene, camera, renderer, meshes, cam,
    project(idx) {
      const n = nodes[idx];
      const v = new THREE.Vector3(n.x3, n.y3, n.z3).project(camera);
      const r = renderer.domElement.getBoundingClientRect();
      const hostR = host.getBoundingClientRect();
      return {
        x: ((v.x + 1) / 2) * r.width + (r.left - hostR.left),
        y: ((-v.y + 1) / 2) * r.height + (r.top - hostR.top),
      };
    },
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.domElement.replaceWith(renderer.domElement.cloneNode(false));
      renderer.dispose();
      meshes.forEach(m => { m.geometry.dispose(); m.material.dispose(); });
      labelSprites.forEach(s => {
        if (s.material.map) s.material.map.dispose();
        s.material.dispose();
      });
      lineGeo.dispose(); lineMat.dispose();
      host.innerHTML = "";
    },
    highlight(idx) {
      const conn = new Set();
      if (idx != null) {
        conn.add(idx);
        __kbNet.edges.forEach(e => {
          if (e.a === idx) conn.add(e.b);
          if (e.b === idx) conn.add(e.a);
        });
      }
      meshes.forEach((mesh, i) => {
        const dim = idx != null && !conn.has(i);
        mesh.material.opacity = dim ? 0.18 : 1;
        mesh.material.transparent = dim;
        mesh.scale.setScalar(i === idx ? 1.35 : 1);
      });
      labelSprites.forEach((sp, i) => {
        // Hide the selected node's own label so it doesn't fight with the
        // floating description card that anchors above the same sphere.
        if (idx != null && i === idx) {
          sp.visible = false;
        } else {
          sp.visible = true;
          const dim = idx != null && !conn.has(i);
          sp.material.opacity = dim ? 0.15 : 1;
        }
      });
      // Highlight edges by rebuilding the line color buffer is overkill;
      // instead toggle global line opacity when something is selected.
      lineMat.opacity = idx != null ? 0.12 : 0.35;
    },
  };

  // Re-bind shared interaction wiring (no-op if already wired).
  _setupNetInteractions();

  // Restore selection.
  if (__kbNet.selectedId) {
    const idx = nodes.findIndex(n => n.id === __kbNet.selectedId);
    if (idx >= 0) _selectNode(idx, false);
  }
}

function _kb3dProject(node) {
  const t = __kbNet.three;
  if (!t) return null;
  const idx = __kbNet.nodes.indexOf(node);
  if (idx < 0) return null;
  return t.project(idx);
}

function _kb3dResetCamera() {
  const t = __kbNet.three;
  if (!t) return;
  t.cam.yaw = 0; t.cam.pitch = 0; t.cam.dist = 600;
  t.cam.target.set(0, 0, 0);
  // Force redraw of the camera.
  const cp = Math.cos(t.cam.pitch), sp = Math.sin(t.cam.pitch);
  const cy = Math.cos(t.cam.yaw), sy = Math.sin(t.cam.yaw);
  t.camera.position.set(t.cam.dist * cp * sy, t.cam.dist * sp, t.cam.dist * cp * cy);
  t.camera.lookAt(t.cam.target);
}

// Smoothly fly the camera so the picked node + its direct neighbors fill the view.
function _kb3dZoomToNeighborhood(idx) {
  const t = __kbNet.three;
  if (!t) return;
  const { THREE, camera, cam } = t;
  const nodes = __kbNet.nodes;
  const n = nodes[idx];
  if (!n) return;
  // Collect the node + its neighbors.
  const ids = new Set([idx]);
  __kbNet.edges.forEach(e => {
    if (e.a === idx) ids.add(e.b);
    if (e.b === idx) ids.add(e.a);
  });
  // Bounding sphere of the cluster.
  const center = new THREE.Vector3();
  const points = [];
  ids.forEach(i => {
    const p = nodes[i];
    if (!p) return;
    const v = new THREE.Vector3(p.x3, p.y3, p.z3);
    points.push(v); center.add(v);
  });
  if (!points.length) return;
  center.multiplyScalar(1 / points.length);
  let maxR = 0;
  points.forEach(p => { maxR = Math.max(maxR, p.distanceTo(center)); });
  // Radius padding so neighbors aren't clipped at the edges.
  const radius = Math.max(maxR + 30, 60);
  // Distance such that the sphere fits in the camera vertical FOV.
  const fov = camera.fov * Math.PI / 180;
  const fitDist = (radius / Math.sin(fov / 2)) * 1.05;
  // Animate yaw/pitch toward the cluster center, dist toward fitDist.
  const startYaw = cam.yaw, startPitch = cam.pitch, startDist = cam.dist;
  const startTarget = cam.target.clone();
  // Compute end yaw/pitch so the camera looks at center from a similar angle
  // but within reasonable bounds.
  const targetYaw = Math.atan2(center.x, center.z) || startYaw;
  const len = Math.hypot(center.x, center.z) || 1;
  const targetPitch = Math.atan2(center.y, len);
  const t0 = performance.now();
  const dur = 600;
  if (__kbNet._zoomRaf) cancelAnimationFrame(__kbNet._zoomRaf);
  const ease = (k) => k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
  const step = () => {
    const k = Math.min(1, (performance.now() - t0) / dur);
    const e = ease(k);
    cam.yaw   = startYaw   + (targetYaw   - startYaw)   * e;
    cam.pitch = Math.max(-1.4, Math.min(1.4, startPitch + (targetPitch - startPitch) * e));
    cam.dist  = startDist  + (fitDist     - startDist)  * e;
    cam.target.lerpVectors(startTarget, center, e);
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const cy = Math.cos(cam.yaw),   sy = Math.sin(cam.yaw);
    camera.position.set(
      cam.target.x + cam.dist * cp * sy,
      cam.target.y + cam.dist * sp,
      cam.target.z + cam.dist * cp * cy,
    );
    camera.lookAt(cam.target);
    if (__kbNet.selectedId) _positionCardFromSelected();
    if (k < 1) {
      __kbNet._zoomRaf = requestAnimationFrame(step);
    } else {
      __kbNet._zoomRaf = 0;
    }
  };
  step();
  // Surface the focused node's card.
  _selectNode(idx, false);
}

// ---------------- Knowledge drawer (chunk browser) ----------------
async function openKbDrawer(card) {
  const drawer = document.getElementById("kb-drawer");
  if (!drawer) return;
  drawer.classList.add("open");
  drawer.hidden = false;
  const titleEl = document.getElementById("kb-drawer-title");
  const metaEl = document.getElementById("kb-drawer-meta");
  const bodyEl = document.getElementById("kb-drawer-body");
  if (titleEl) titleEl.textContent = card.title || card.doc_id || "(untitled)";
  if (metaEl) {
    const ts = card.ingested_at ? new Date(card.ingested_at * 1000).toLocaleString() : "";
    const tagsHtml = (card.tags || []).map(t => `<span class="kb-tag">${escapeHtml(t)}</span>`).join("");
    metaEl.innerHTML = `
      <div><strong>source:</strong> ${escapeHtml(card.source || "(none)")}</div>
      <div><strong>doc_id:</strong> ${escapeHtml(card.doc_id || "")}</div>
      <div><strong>chunks:</strong> ${card.chunks}</div>
      ${ts ? `<div><strong>ingested:</strong> ${ts}</div>` : ""}
      <div class="kb-card-tags">${tagsHtml}</div>
    `;
  }
  if (bodyEl) bodyEl.innerHTML = `<div class="kb-empty">loading chunks…</div>`;
  try {
    const r = await fetch("/rag/chunks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids: card.chunk_ids || [] }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    const chunks = j.chunks || [];
    if (!chunks.length) {
      bodyEl.innerHTML = `<div class="kb-empty">No chunks.</div>`;
      return;
    }
    bodyEl.innerHTML = "";
    for (const c of chunks) {
      const div = document.createElement("div");
      div.className = "kb-chunk";
      const head = document.createElement("div");
      head.className = "kb-chunk-head";
      head.innerHTML = `#${c.chunk_index ?? "?"} <span class="kb-chunk-id"></span>`;
      head.querySelector(".kb-chunk-id").textContent = c.id || "";
      const md = document.createElement("div");
      md.className = "kb-chunk-text markdown";
      md.innerHTML = renderMarkdown(c.text || "");
      enhanceCodeBlocks(md);
      renderMath(md);
      div.appendChild(head);
      div.appendChild(md);
      bodyEl.appendChild(div);
    }
  } catch (e) {
    bodyEl.innerHTML = `<div class="kb-empty">load failed: ${escapeHtml(e.message || String(e))}</div>`;
  }
}

function closeKbDrawer() {
  const drawer = document.getElementById("kb-drawer");
  if (!drawer) return;
  drawer.classList.remove("open");
  setTimeout(() => { drawer.hidden = true; }, 200);
}

// ---------------- Activity filters & stats ----------------
function applyActivityFilters() {
  const txt = (document.getElementById("act-filter-text")?.value || "").toLowerCase().trim();
  const status = document.getElementById("act-filter-status")?.value || "";
  const model = document.getElementById("act-filter-model")?.value || "";
  const list = document.getElementById("activity-list");
  if (!list) return;
  let visible = 0;
  list.querySelectorAll(".act-card").forEach(card => {
    const cs = card.dataset.status || "live";
    const cm = card.dataset.model || "";
    const cc = card.dataset.client || "";
    const cp = (card.dataset.preview || "").toLowerCase();
    let hide = false;
    if (status && cs !== status) hide = true;
    if (model && cm !== model) hide = true;
    if (txt && !(cp.includes(txt) || cm.toLowerCase().includes(txt) || cc.toLowerCase().includes(txt))) hide = true;
    card.classList.toggle("hidden-by-filter", hide);
    if (!hide) visible++;
  });
  const empty = document.getElementById("activity-empty");
  if (empty) empty.hidden = visible > 0 || list.children.length === 0 ? empty.hidden : false;
}

function refreshActivityStats() {
  const live = document.getElementById("act-stat-live");
  const total = document.getElementById("act-stat-total");
  const err = document.getElementById("act-stat-err");
  const tps = document.getElementById("act-stat-tps");
  if (live) live.textContent = String(activity.liveCount);
  if (total) total.textContent = String(activity.total);
  if (err) err.textContent = String(activity.errors);
  if (tps) {
    const samples = activity.tpsSamples || [];
    if (samples.length) {
      const avg = samples.reduce((a, b) => a + b, 0) / samples.length;
      tps.textContent = avg.toFixed(1);
    } else {
      tps.textContent = "—";
    }
  }
}

function refreshActivityModelFilter() {
  const sel = document.getElementById("act-filter-model");
  if (!sel) return;
  const cur = sel.value;
  const models = Array.from(activity.modelSet || []).sort();
  sel.innerHTML = `<option value="">all models</option>` +
    models.map(m => `<option value="${escapeAttr(m)}">${escapeHtml(m)}</option>`).join("");
  if (cur && models.includes(cur)) sel.value = cur;
}

bootstrap();
