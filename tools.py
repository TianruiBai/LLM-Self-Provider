"""Built-in tools available to chat models.

Two tools are exposed as OpenAI function definitions:

  * ``web_search(query, max_results=5)`` — Tavily first, DuckDuckGo HTML scrape
    fallback when ``TAVILY_API_KEY`` is unset or the API errors.
  * ``web_fetch(url, max_chars=8000)`` — fetches a URL and returns the
    visible text (HTML stripped).

The chat gateway auto-executes these functions when an upstream response
emits matching ``tool_calls``; the result is appended as a ``role="tool"``
message and the conversation is re-sent until the model replies without
any tool_call (capped at 4 hops to keep latency bounded).

Whisper transcription lives at the bottom of this module — model is loaded
lazily on first call and pinned to CPU so it never competes with chat VRAM.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.parse
from html.parser import HTMLParser
from typing import Any, Iterable

import httpx

log = logging.getLogger("provider.tools")


# ----------------------------- tool catalog -----------------------------

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web for up-to-date information. Returns the "
            "top results with title, url, and a short snippet. Use this when "
            "the user asks about recent events, live data, or anything you "
            "are not confident about."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "How many results to return (default 5, max 10).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}

WEB_FETCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "Fetch a URL and return its readable text content. Use after a "
            "web_search to read the full body of a promising page. Will "
            "follow redirects and strip HTML/scripts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute http(s) URL to fetch."},
                "max_chars": {
                    "type": "integer",
                    "description": "Truncate the returned text to this many chars (default 8000).",
                    "minimum": 200,
                    "maximum": 32000,
                },
            },
            "required": ["url"],
        },
    },
}

# Document tools — registered only when the chat request carries attached
# documents. They give the model the ability to list, read (paginated) and
# keyword-search the user's uploaded files instead of forcing the entire text
# into the prompt context.
LIST_DOCUMENTS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_documents",
        "description": (
            "List the documents the user attached to this conversation. "
            "Returns each document's id, name, format and total length. "
            "Call this first to discover what is available, then use "
            "`read_document` or `search_documents` to inspect them."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

READ_DOCUMENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_document",
        "description": (
            "Return a slice of an attached document's plain-text content. "
            "Use after `list_documents` to fetch the body. Supports "
            "pagination via `offset`/`max_chars` for large files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document id from list_documents."},
                "offset": {"type": "integer", "description": "Start character offset (default 0).", "minimum": 0},
                "max_chars": {"type": "integer", "description": "Maximum chars to return (default 8000, max 32000).", "minimum": 200, "maximum": 32000},
            },
            "required": ["id"],
        },
    },
}

SEARCH_DOCUMENTS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Case-insensitive keyword search across all attached documents. "
            "Returns up to `max_hits` matched snippets with the document id, "
            "name, char offset and surrounding context. Each space-separated "
            "term must appear in the snippet (AND search)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords (space-separated, AND)."},
                "max_hits": {"type": "integer", "description": "Max snippets to return (default 8, max 20).", "minimum": 1, "maximum": 20},
                "context_chars": {"type": "integer", "description": "Chars of context on each side of a hit (default 160).", "minimum": 40, "maximum": 800},
                "id": {"type": "string", "description": "Optional: search only this document id."},
            },
            "required": ["query"],
        },
    },
}

DOC_TOOLS: list[dict[str, Any]] = [LIST_DOCUMENTS_TOOL, READ_DOCUMENT_TOOL, SEARCH_DOCUMENTS_TOOL]
DOC_NAMES: set[str] = {t["function"]["name"] for t in DOC_TOOLS}

BUILTIN_TOOLS: list[dict[str, Any]] = [WEB_SEARCH_TOOL, WEB_FETCH_TOOL]
BUILTIN_NAMES: set[str] = {t["function"]["name"] for t in BUILTIN_TOOLS} | DOC_NAMES


# --------------------------- document tool impl ---------------------------


def _docs_index(docs: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, d in enumerate(docs or []):
        if not isinstance(d, dict):
            continue
        did = str(d.get("id") or f"doc{i + 1}")
        out[did] = {
            "id": did,
            "name": d.get("name") or did,
            "format": d.get("format") or "text",
            "text": d.get("text") or "",
            "size": int(d.get("size") or len(d.get("text") or "")),
        }
    return out


def list_documents(docs: list[dict[str, Any]] | None) -> dict[str, Any]:
    idx = _docs_index(docs)
    if not idx:
        return {"documents": [], "note": "No documents are attached to this conversation."}
    return {
        "documents": [
            {"id": d["id"], "name": d["name"], "format": d["format"], "length": len(d["text"])}
            for d in idx.values()
        ],
    }


def read_document(
    docs: list[dict[str, Any]] | None,
    *,
    id: str,
    offset: int = 0,
    max_chars: int = 8000,
) -> dict[str, Any]:
    idx = _docs_index(docs)
    d = idx.get(str(id))
    if not d:
        return {"error": f"document {id!r} not found", "available": list(idx.keys())}
    text = d["text"] or ""
    total = len(text)
    offset = max(0, min(int(offset or 0), total))
    max_chars = max(200, min(int(max_chars or 8000), 32000))
    chunk = text[offset : offset + max_chars]
    return {
        "id": d["id"],
        "name": d["name"],
        "format": d["format"],
        "offset": offset,
        "length": len(chunk),
        "total": total,
        "truncated": (offset + len(chunk)) < total,
        "text": chunk,
    }


def search_documents(
    docs: list[dict[str, Any]] | None,
    *,
    query: str,
    max_hits: int = 8,
    context_chars: int = 160,
    id: str | None = None,
) -> dict[str, Any]:
    idx = _docs_index(docs)
    if not idx:
        return {"hits": [], "note": "No documents are attached."}
    if id:
        target = idx.get(str(id))
        items = [target] if target else []
        if not items:
            return {"error": f"document {id!r} not found", "available": list(idx.keys())}
    else:
        items = list(idx.values())
    terms = [t.lower() for t in (query or "").split() if t.strip()]
    if not terms:
        return {"hits": [], "note": "Empty query."}
    primary = terms[0]
    max_hits = max(1, min(int(max_hits or 8), 20))
    context_chars = max(40, min(int(context_chars or 160), 800))
    hits: list[dict[str, Any]] = []
    for d in items:
        text = d["text"] or ""
        low = text.lower()
        start = 0
        while len(hits) < max_hits:
            pos = low.find(primary, start)
            if pos == -1:
                break
            seg_start = max(0, pos - context_chars)
            seg_end = min(len(text), pos + len(primary) + context_chars)
            snippet = text[seg_start:seg_end]
            if all(t in snippet.lower() for t in terms):
                hits.append({
                    "id": d["id"],
                    "name": d["name"],
                    "offset": pos,
                    "snippet": snippet,
                })
            start = pos + max(1, len(primary))
        if len(hits) >= max_hits:
            break
    return {"hits": hits, "query": query, "matched_terms": terms}


# --------------------------- HTML to plain text ---------------------------


class _TextExtractor(HTMLParser):
    """Best-effort HTML -> readable text without external deps."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "header", "footer", "nav", "aside", "form"}
    _BLOCK_TAGS = {
        "p", "br", "div", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
        "pre", "blockquote", "section", "article", "hr",
    }

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if data:
            self._chunks.append(data)

    def text(self) -> str:
        joined = "".join(self._chunks)
        # Collapse internal whitespace within lines, keep paragraph breaks.
        joined = re.sub(r"[ \t\f\v]+", " ", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        return joined.strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception as e:  # noqa: BLE001
        log.debug("html_to_text parser failed: %s", e)
    return parser.text()


# ----------------------------- web_search -----------------------------


async def web_search(query: str, max_results: int = 5, *, http: httpx.AsyncClient | None = None) -> dict[str, Any]:
    query = (query or "").strip()
    if not query:
        return {"error": "empty query", "results": []}
    max_results = max(1, min(10, int(max_results)))
    own_client = http is None
    client = http or httpx.AsyncClient(
        timeout=httpx.Timeout(20.0),
        follow_redirects=True,
        headers={"User-Agent": _user_agent()},
    )
    try:
        # Try Tavily first (rich snippets, low latency).
        api_key = os.environ.get("TAVILY_API_KEY")
        if api_key:
            try:
                results = await _tavily_search(client, api_key, query, max_results)
                if results:
                    return {"query": query, "provider": "tavily", "results": results}
            except Exception as e:  # noqa: BLE001
                log.warning("Tavily failed, falling back to DDG: %s", e)
        # Fallback: DuckDuckGo HTML.
        try:
            results = await _ddg_search(client, query, max_results)
            return {"query": query, "provider": "duckduckgo", "results": results}
        except Exception as e:  # noqa: BLE001
            return {"query": query, "provider": "none", "results": [], "error": str(e)}
    finally:
        if own_client:
            await client.aclose()


async def _tavily_search(client: httpx.AsyncClient, api_key: str, query: str, max_results: int) -> list[dict[str, Any]]:
    r = await client.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        },
        timeout=20.0,
    )
    r.raise_for_status()
    payload = r.json()
    out = []
    for item in payload.get("results", [])[:max_results]:
        out.append({
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "snippet": (item.get("content") or "")[:400],
        })
    return out


async def _ddg_search(client: httpx.AsyncClient, query: str, max_results: int) -> list[dict[str, Any]]:
    # The HTML endpoint is intentionally simple to scrape.
    r = await client.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query, "kl": "us-en"},
        timeout=20.0,
    )
    r.raise_for_status()
    html = r.text
    # Parse <a class="result__a" href="..."> ... </a> + <a class="result__snippet">.
    out: list[dict[str, Any]] = []
    block_re = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?(?:<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>)?',
        re.IGNORECASE | re.DOTALL,
    )
    for m in block_re.finditer(html):
        href = _unwrap_ddg_redirect(m.group(1))
        title = html_to_text(m.group(2))
        snippet = html_to_text(m.group(3) or "")[:400]
        if href.startswith("http"):
            out.append({"title": title, "url": href, "snippet": snippet})
        if len(out) >= max_results:
            break
    return out


def _unwrap_ddg_redirect(href: str) -> str:
    # DDG html results sometimes wrap real urls in /l/?uddg=...
    if href.startswith("//"):
        href = "https:" + href
    if "duckduckgo.com/l/" in href:
        try:
            qs = urllib.parse.urlparse(href).query
            real = urllib.parse.parse_qs(qs).get("uddg", [None])[0]
            if real:
                return urllib.parse.unquote(real)
        except Exception:
            pass
    return href


# ----------------------------- web_fetch -----------------------------


async def web_fetch(url: str, max_chars: int = 8000, *, http: httpx.AsyncClient | None = None) -> dict[str, Any]:
    url = (url or "").strip()
    if not url:
        return {"error": "empty url"}
    if not (url.startswith("http://") or url.startswith("https://")):
        return {"error": "url must start with http:// or https://"}
    max_chars = max(200, min(32000, int(max_chars)))
    own_client = http is None
    client = http or httpx.AsyncClient(
        timeout=httpx.Timeout(25.0),
        follow_redirects=True,
        headers={"User-Agent": _user_agent()},
    )
    try:
        r = await client.get(url)
        ctype = r.headers.get("content-type", "").lower()
        body: str
        if "text/html" in ctype or "application/xhtml" in ctype or (not ctype and r.text):
            body = html_to_text(r.text)
        elif ctype.startswith("text/") or "json" in ctype or "xml" in ctype:
            body = r.text
        else:
            return {
                "url": str(r.url),
                "status": r.status_code,
                "content_type": ctype,
                "error": "non-text content; refusing to dump bytes",
            }
        truncated = len(body) > max_chars
        return {
            "url": str(r.url),
            "status": r.status_code,
            "content_type": ctype,
            "title": _extract_title(r.text) if "html" in ctype else None,
            "text": body[:max_chars],
            "truncated": truncated,
            "length": len(body),
        }
    except httpx.HTTPError as e:
        return {"url": url, "error": f"http error: {e}"}
    finally:
        if own_client:
            await client.aclose()


def _extract_title(html: str) -> str | None:
    m = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return html_to_text(m.group(1))[:200]


def _user_agent() -> str:
    return os.environ.get(
        "PROVIDER_USER_AGENT",
        "Mozilla/5.0 (compatible; SelfHostedProvider/0.1; +local)",
    )


# ----------------------------- dispatch -----------------------------


async def execute_tool(
    name: str,
    args: dict[str, Any],
    *,
    http: httpx.AsyncClient | None = None,
    docs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute a built-in tool by name. Always returns a JSON-serializable dict."""
    try:
        if name == "web_search":
            return await web_search(
                str(args.get("query", "")),
                int(args.get("max_results", 5) or 5),
                http=http,
            )
        if name == "web_fetch":
            return await web_fetch(
                str(args.get("url", "")),
                int(args.get("max_chars", 8000) or 8000),
                http=http,
            )
        if name == "list_documents":
            return list_documents(docs)
        if name == "read_document":
            return read_document(
                docs,
                id=str(args.get("id", "")),
                offset=int(args.get("offset", 0) or 0),
                max_chars=int(args.get("max_chars", 8000) or 8000),
            )
        if name == "search_documents":
            return search_documents(
                docs,
                query=str(args.get("query", "")),
                max_hits=int(args.get("max_hits", 8) or 8),
                context_chars=int(args.get("context_chars", 160) or 160),
                id=(str(args["id"]) if args.get("id") else None),
            )
        return {"error": f"unknown built-in tool: {name}"}
    except Exception as e:  # noqa: BLE001
        log.exception("tool %s failed", name)
        return {"error": str(e)}


def merge_tools(
    user_tools: list[dict[str, Any]] | None,
    want_builtin: bool,
    *,
    has_documents: bool = False,
) -> list[dict[str, Any]]:
    """Merge user-provided tool list with the built-in catalog when requested.

    When ``has_documents`` is True, the document tools are appended even if
    the caller hasn't enabled the web tool catalog, because they only need
    the per-request docs context (no external network).
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    if user_tools:
        for t in user_tools:
            try:
                name = t["function"]["name"]
            except Exception:
                continue
            seen.add(name)
            out.append(t)
    if want_builtin:
        for t in BUILTIN_TOOLS:
            if t["function"]["name"] not in seen:
                out.append(t)
                seen.add(t["function"]["name"])
    if has_documents:
        for t in DOC_TOOLS:
            if t["function"]["name"] not in seen:
                out.append(t)
    return out


def is_builtin(name: str) -> bool:
    return name in BUILTIN_NAMES


# ----------------------------- whisper transcription -----------------------------


_WHISPER_MODEL: Any = None
_WHISPER_LOCK = asyncio.Lock()


async def transcribe_audio(audio_bytes: bytes, *, model_name: str = "base", language: str | None = None) -> dict[str, Any]:
    """Transcribe an audio blob using openai-whisper (CPU).

    The model is cached after the first call. The function offloads the
    blocking decode to a worker thread so the gateway event loop stays free.
    """
    if not audio_bytes:
        return {"error": "empty audio"}
    try:
        import whisper  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        return {
            "error": (
                "openai-whisper is not installed in this environment. "
                f"Original import error: {e}"
            )
        }

    global _WHISPER_MODEL
    async with _WHISPER_LOCK:
        if _WHISPER_MODEL is None:
            log.info("Loading Whisper model %r (CPU, first call) ...", model_name)
            _WHISPER_MODEL = await asyncio.to_thread(whisper.load_model, model_name, "cpu")

    import tempfile

    def _decode() -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        try:
            result = _WHISPER_MODEL.transcribe(path, language=language, fp16=False)
            return {
                "text": (result.get("text") or "").strip(),
                "language": result.get("language"),
                "duration": result.get("duration"),
            }
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    return await asyncio.to_thread(_decode)


__all__ = [
    "BUILTIN_TOOLS",
    "BUILTIN_NAMES",
    "execute_tool",
    "merge_tools",
    "is_builtin",
    "web_search",
    "web_fetch",
    "transcribe_audio",
]
