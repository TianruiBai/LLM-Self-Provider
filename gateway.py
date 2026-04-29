"""FastAPI gateway exposing OpenAI-compatible endpoints."""
from __future__ import annotations

import json
import logging
import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .events import EventBus
from .lifecycle import LifecycleManager
from .rag import RagService
from .registry import ProviderConfig, load_config
from . import tools as builtin_tools
from . import downloader as model_downloader

log = logging.getLogger("provider.gateway")


# ----------------- request models (lenient: extra fields pass through) -----------------


class RagOptions(BaseModel):
    enabled: bool = True
    top_k: int | None = None
    source: str | None = None
    tags: list[str] | None = None
    query: str | None = None  # override; otherwise last user message is used


class IngestDoc(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDoc]
    source: str = "api"
    tags: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    text: str
    top_k: int | None = None
    source: str | None = None
    tags: list[str] | None = None


# ----------------- app factory -----------------


def _replay_as_sse(payload: dict[str, Any], model_id: str) -> AsyncIterator[bytes]:
    """Convert a non-streamed completion payload into an SSE event stream.

    Used after the tool-call loop when the original request asked for
    `stream: true`: the gateway has already consumed the upstream response
    non-streaming (so it could inspect tool_calls), so we replay the final
    assistant content as a tiny SSE stream the client expects.
    """
    async def gen() -> AsyncIterator[bytes]:
        choice = (payload.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        created = int(time.time())
        cid = "chatcmpl-" + uuid.uuid4().hex[:12]

        def _event(delta: dict[str, Any], finish: str | None = None) -> bytes:
            chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": payload.get("model") or model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(chunk)}\n\n".encode("utf-8")

        yield _event({"role": "assistant"})
        if reasoning:
            # Stream reasoning in one shot for compatibility.
            yield _event({"reasoning_content": reasoning})
        # Chunk the content into ~256-char pieces so clients render progressively.
        step = 256
        if content:
            for i in range(0, len(content), step):
                yield _event({"content": content[i : i + step]})
        # Final usage frame, then the [DONE] sentinel.
        usage = payload.get("usage")
        final = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": payload.get("model") or model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason") or "stop"}],
        }
        if usage:
            final["usage"] = usage
        yield f"data: {json.dumps(final)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return gen()


def create_app(cfg: ProviderConfig | None = None) -> FastAPI:
    cfg = cfg or load_config()
    lifecycle = LifecycleManager(cfg)
    rag = RagService(cfg, lifecycle.ensure_embedder)
    bus = EventBus()

    # Load persisted runtime config (e.g. user-supplied API keys) and mirror it
    # into ``os.environ`` so existing helpers like tools.web_search pick it up.
    from provider import runtime_config
    runtime_config.load()

    # Initialize control-plane SQLite DB (users, sessions, API keys, audit).
    # This is a synchronous, fast, idempotent call (creates ~7 tables on first
    # run, no-op afterwards) so we run it eagerly before the app starts.
    from provider import db as _control_db
    try:
        _control_db.init()
    except Exception as e:  # noqa: BLE001
        log.exception("Control DB init failed: %s", e)
        raise

    # Optional: create the first admin from environment variables on startup.
    # Skipped silently if any user already exists.
    _bootstrap_user = os.environ.get("PROVIDER_BOOTSTRAP_ADMIN_USER")
    _bootstrap_pw = os.environ.get("PROVIDER_BOOTSTRAP_ADMIN_PASSWORD")
    if _bootstrap_user and _bootstrap_pw:
        try:
            from provider import auth as _auth_svc
            existing = _control_db.fetchone("SELECT COUNT(*) AS n FROM users")
            if existing and existing["n"] == 0:
                _auth_svc.ensure_initial_admin(_bootstrap_user, _bootstrap_pw)
                log.info("Bootstrapped admin user %r from env", _bootstrap_user)
        except Exception as e:  # noqa: BLE001
            log.exception("Admin bootstrap failed: %s", e)


    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        log.info("Starting provider service")
        # Hook the idle watchdog into the activity event bus so the UI
        # learns when CUDA0 helpers get auto-evicted.
        def _on_idle_unload(kind: str, model_id: str) -> None:
            evt = "embedder.unloaded" if kind == "embedding" else (
                "vision.unloaded" if kind == "vision" else "sub_agent.unloaded"
            )
            bus.publish({"type": evt, "model": model_id, "reason": "idle"})
        lifecycle.set_idle_callback(_on_idle_unload)
        await lifecycle.startup()
        try:
            await rag.startup()
        except Exception as e:  # noqa: BLE001
            log.warning("RAG startup failed (continuing without RAG): %s", e)
        try:
            yield
        finally:
            log.info("Stopping provider service")
            await rag.shutdown()
            await lifecycle.shutdown()

    app = FastAPI(title="Self-hosted Model Provider", version="0.1.0", lifespan=lifespan)

    # CORS — allow OpenAI clients from anywhere (VS Code webviews, browser tools, etc.)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Lightweight request logger.  We deliberately implement this as a *pure
    # ASGI* middleware (not ``@app.middleware("http")``) because Starlette's
    # ``BaseHTTPMiddleware`` buffers ``StreamingResponse`` bodies to completion
    # before forwarding them — which kills SSE streaming for chat completions.
    class _AccessLogMiddleware:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope.get("type") != "http":
                await self.app(scope, receive, send)
                return
            t0 = time.time()
            status_holder = {"code": 0}
            path = scope.get("path", "")
            method = scope.get("method", "")

            async def send_wrapper(message):
                if message.get("type") == "http.response.start":
                    status_holder["code"] = int(message.get("status", 0))
                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
            except Exception:
                log.exception("Unhandled error: %s %s", method, path)
                raise
            finally:
                dt = (time.time() - t0) * 1000
                if not path.startswith("/ui/") and path not in ("/health",):
                    log.info("%s %s -> %d (%.0f ms)", method, path, status_holder["code"], dt)

    app.add_middleware(_AccessLogMiddleware)

    # Authentication middleware: resolves the caller (Bearer API key or
    # session cookie) onto ``request.state.actor`` and rejects unauthenticated
    # access to protected route prefixes (/v1/*, /admin/*, /rag/*, /events,
    # /auth/me|logout|keys|totp|users|sessions). Public prefixes (/health,
    # /ui/, /auth/login, /auth/bootstrap) and unprotected legacy routes are
    # passed through untouched.
    from provider.auth_deps import AuthMiddleware
    app.add_middleware(AuthMiddleware)

    # /auth router: login/logout/me, API-key CRUD, TOTP, recovery codes,
    # bootstrap, admin user/session/key management.
    from provider.auth_routes import router as _auth_router
    app.include_router(_auth_router)

    proxy_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None))

    # ---------------- health ----------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "active_chat_model": lifecycle.active_chat_model(),
            "active_sub_agent_model": lifecycle.active_sub_agent_model(),
            "active_vision_model": lifecycle.active_vision_model(),
            "embedder_ready": lifecycle.embedding_base_url() is not None,
            "sub_agent_ready": lifecycle.sub_agent_base_url() is not None,
            "vision_ready": lifecycle.vision_base_url() is not None,
        }

    @app.get("/admin/status")
    async def admin_status() -> dict[str, Any]:
        return {
            "active_chat_model": lifecycle.active_chat_model(),
            "active_sub_agent_model": lifecycle.active_sub_agent_model(),
            "active_vision_model": lifecycle.active_vision_model(),
            "embedder_ready": lifecycle.embedding_base_url() is not None,
            "sub_agent_ready": lifecycle.sub_agent_base_url() is not None,
            "vision_ready": lifecycle.vision_base_url() is not None,
            "models": [
                {
                    "id": m.id,
                    "kind": m.kind,
                    "path": m.path,
                    "path_exists": Path(m.path).exists(),
                    "mmproj": m.mmproj,
                    "mmproj_exists": (Path(m.mmproj).exists() if m.mmproj else None),
                    "folder": m.folder,
                    "download": m.download,
                }
                for m in cfg.models
            ],
        }

    @app.post("/admin/unload")
    async def admin_unload() -> dict[str, Any]:
        unloaded = await lifecycle.unload_chat()
        bus.publish({"type": "model.unloaded", "model": unloaded})
        return {"unloaded": unloaded, "active_chat_model": lifecycle.active_chat_model()}

    @app.post("/admin/unload-embedder")
    async def admin_unload_embedder() -> dict[str, Any]:
        unloaded = await lifecycle.unload_embedder()
        bus.publish({"type": "embedder.unloaded", "model": unloaded})
        return {"unloaded": unloaded}

    @app.post("/admin/unload-sub-agent")
    async def admin_unload_sub_agent() -> dict[str, Any]:
        unloaded = await lifecycle.unload_sub_agent()
        bus.publish({"type": "sub_agent.unloaded", "model": unloaded})
        return {"unloaded": unloaded}

    @app.post("/admin/unload-vision")
    async def admin_unload_vision() -> dict[str, Any]:
        unloaded = await lifecycle.unload_vision()
        bus.publish({"type": "vision.unloaded", "model": unloaded})
        return {"unloaded": unloaded}

    # ---------------- model download ----------------

    # Track in-flight downloads so we can refuse double-starts and report
    # status via /admin/fetch-model/status.
    _downloads: dict[str, dict[str, Any]] = {}

    @app.get("/admin/fetch-model/status")
    async def admin_fetch_status() -> dict[str, Any]:
        return {"downloads": _downloads}

    @app.post("/admin/fetch-model")
    async def admin_fetch_model(payload: dict) -> dict[str, Any]:
        """Download a model's GGUF (and optional mmproj) into its folder.

        Body: {model: "<id>"}. The model must declare a `download:` block in
        its `model.yaml`. Progress is published on the EventBus so the web UI
        can render a progress bar via /events.
        """
        model_id = (payload.get("model") or "").strip()
        if not model_id:
            raise HTTPException(status_code=400, detail="`model` is required")
        m = next((mm for mm in cfg.models if mm.id == model_id), None)
        if m is None:
            raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")
        if not m.download:
            raise HTTPException(status_code=400, detail=f"model {model_id!r} has no `download:` block in its model.yaml")
        spec = m.download
        repo = (spec.get("repo") or "").strip()
        gguf_file = (spec.get("file") or "").strip()
        mmproj_file = (spec.get("mmproj_file") or "").strip() or None
        revision = (spec.get("revision") or None)
        if not repo or not gguf_file:
            raise HTTPException(status_code=400, detail="download spec requires `repo` and `file`")

        target_dir = Path(m.folder) if m.folder else Path(m.path).parent
        if model_id in _downloads and _downloads[model_id].get("phase") not in ("done", "error", "skip"):
            raise HTTPException(status_code=409, detail=f"download already in progress for {model_id}")

        state: dict[str, Any] = {
            "model": model_id, "phase": "queued",
            "downloaded": 0, "total": None,
            "started_at": time.time(),
        }
        _downloads[model_id] = state

        async def _emit(ev: dict[str, Any]) -> None:
            state.update(ev)
            bus.publish({"type": "model.download", "model": model_id, **ev})

        async def _run() -> None:
            try:
                # Main GGUF.
                final_name = Path(m.path).name
                bus.publish({"type": "model.download", "model": model_id, "phase": "begin", "file": gguf_file})
                await model_downloader.fetch_hf_file(
                    repo=repo, filename=gguf_file, revision=revision,
                    target_dir=target_dir, progress=_emit, final_name=final_name,
                )
                # Optional mmproj.
                if mmproj_file:
                    bus.publish({"type": "model.download", "model": model_id, "phase": "begin", "file": mmproj_file})
                    final_mmproj = Path(m.mmproj).name if m.mmproj else Path(mmproj_file).name
                    await model_downloader.fetch_hf_file(
                        repo=repo, filename=mmproj_file, revision=revision,
                        target_dir=target_dir, progress=_emit, final_name=final_mmproj,
                    )
                state["phase"] = "done"
                bus.publish({"type": "model.download", "model": model_id, "phase": "complete"})
            except Exception as e:  # noqa: BLE001
                state["phase"] = "error"
                state["error"] = str(e)
                bus.publish({"type": "model.download", "model": model_id, "phase": "error", "error": str(e)})

        asyncio.create_task(_run())
        return {"started": True, "model": model_id, "target_dir": str(target_dir)}

    @app.get("/admin/runtime-config")
    async def admin_runtime_config_get() -> dict[str, Any]:
        from provider import runtime_config
        return runtime_config.get_all()

    @app.post("/admin/runtime-config")
    async def admin_runtime_config_set(payload: dict[str, Any]) -> dict[str, Any]:
        from provider import runtime_config
        return runtime_config.update(payload or {})

    @app.post("/admin/extract")
    async def admin_extract(file: UploadFile = File(...)) -> dict[str, Any]:
        """Extract plain text from an uploaded document (pdf/docx/pptx/xlsx/csv/text)."""
        from provider import doc_extract  # local import to keep startup light
        try:
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="empty file")
            if len(data) > 50 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="file too large (>50 MiB)")
            result = doc_extract.extract(file.filename or "upload", data)
            return result
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"extract failed: {e}")

    @app.post("/admin/summarize")
    async def admin_summarize(payload: dict) -> dict[str, Any]:
        """Summarize text via the sub-agent and ingest the result into RAG.

        Body: {text, title?, source?, tags?, style?, max_tokens?, temperature?}
        """
        text = (payload.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="`text` is required")
        if cfg.sub_agent_model is None:
            raise HTTPException(status_code=503, detail="No sub_agent model configured")
        title = (payload.get("title") or "summary").strip() or "summary"
        source = (payload.get("source") or "summaries").strip() or "summaries"
        tags = list(payload.get("tags") or ["summary"])
        style = payload.get("style") or "concise"
        unload_after = bool(payload.get("unload_after", True))
        try:
            base = await lifecycle.ensure_sub_agent()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Sub-agent failed to start: {e}")
        sys_prompt = (
            "You are a careful summarizer. Produce a faithful, well-structured summary "
            f"in {style} style. Preserve key facts, named entities, and numeric values. "
            "Use markdown with headings and bullet points. Do not invent facts."
        )
        sub_id = cfg.sub_agent_model.id
        try:
            r = await proxy_client.post(
                f"{base}/v1/chat/completions",
                json={
                    "model": sub_id,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": text},
                    ],
                    "stream": False,
                    "temperature": float(payload.get("temperature", 0.3)),
                    "max_tokens": int(payload.get("max_tokens", 1024)),
                },
                timeout=httpx.Timeout(60.0, read=600.0),
            )
        except Exception as e:  # noqa: BLE001
            if unload_after:
                try:
                    await lifecycle.unload_sub_agent()
                except Exception:  # noqa: BLE001
                    log.warning("auto-unload sub_agent after summarize error failed", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Sub-agent request failed: {e}") from e
        if r.status_code >= 400:
            if unload_after:
                try:
                    await lifecycle.unload_sub_agent()
                except Exception:  # noqa: BLE001
                    log.warning("auto-unload sub_agent failed", exc_info=True)
            raise HTTPException(status_code=r.status_code, detail=r.text)
        j = r.json()
        summary = (j.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        if not summary:
            if unload_after:
                try:
                    await lifecycle.unload_sub_agent()
                except Exception:  # noqa: BLE001
                    log.warning("auto-unload sub_agent failed", exc_info=True)
            raise HTTPException(status_code=502, detail="Sub-agent returned empty summary")
        doc_id = f"{title}-{int(time.time())}"
        try:
            ingest_res = await rag.ingest(
                [{"text": summary, "metadata": {"id": doc_id, "title": title}}],
                source=source,
                tags=tags,
            )
        except Exception as e:  # noqa: BLE001
            if unload_after:
                try:
                    await lifecycle.unload_sub_agent()
                except Exception:  # noqa: BLE001
                    log.warning("auto-unload sub_agent failed", exc_info=True)
            return {"summary": summary, "doc_id": doc_id, "ingested": False, "error": str(e)}
        bus.publish({"type": "summary.ingested", "doc_id": doc_id, "source": source, "title": title})
        if unload_after:
            try:
                await lifecycle.unload_sub_agent()
                bus.publish({"type": "sub_agent.unloaded", "model": sub_id, "reason": "auto_after_summarize"})
            except Exception:  # noqa: BLE001
                log.warning("auto-unload sub_agent after summarize failed", exc_info=True)
        return {"summary": summary, "doc_id": doc_id, "title": title, "source": source, "tags": tags, "ingested": True, "ingest": ingest_res, "unloaded": unload_after}

    # ---------------- live activity stream (SSE) ----------------

    @app.get("/events")
    async def events(_req: Request) -> StreamingResponse:
        q = await bus.subscribe()
        return StreamingResponse(
            bus.stream(q),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ---------------- /v1/models ----------------

    def _models_payload() -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": m.id,
                    "object": "model",
                    "created": now,
                    "owned_by": "self-hosted",
                    "kind": m.kind,
                }
                for m in cfg.models
            ],
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return _models_payload()

    # Some IDE clients omit the /v1 prefix.
    @app.get("/models")
    async def list_models_alias() -> dict[str, Any]:
        return _models_payload()

    # ---------------- /v1/embeddings ----------------

    @app.post("/v1/embeddings")
    async def embeddings(req: Request) -> JSONResponse:
        body = await req.json()
        try:
            base = await lifecycle.ensure_embedder()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Failed to start embedder: {e}")
        # Force the registered embedder model id regardless of what's requested.
        if cfg.embedding_model is not None:
            body["model"] = cfg.embedding_model.id
        r = await proxy_client.post(f"{base}/v1/embeddings", json=body)
        return JSONResponse(status_code=r.status_code, content=r.json())

    # ---------------- /v1/chat/completions ----------------

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        body = await req.json()
        return await _chat_or_completions(req, body, "/v1/chat/completions", chat=True)

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        return await _chat_or_completions(req, body, "/v1/completions", chat=False)

    # Aliases without /v1 prefix for clients that omit it.
    @app.post("/chat/completions")
    async def chat_completions_alias(req: Request):
        body = await req.json()
        return await _chat_or_completions(req, body, "/v1/chat/completions", chat=True)

    @app.post("/completions")
    async def completions_alias(req: Request):
        body = await req.json()
        return await _chat_or_completions(req, body, "/v1/completions", chat=False)

    # Dedicated sub-agent endpoints. Clients may also call /v1/chat/completions
    # directly with the sub_agent model id; both paths reach the same process.
    @app.post("/v1/sub-agent/chat/completions")
    async def sub_agent_chat(req: Request):
        body = await req.json()
        sub = cfg.sub_agent_model
        if sub is None:
            raise HTTPException(status_code=503, detail="No sub_agent model configured")
        body.setdefault("model", sub.id)
        return await _chat_or_completions(req, body, "/v1/chat/completions", chat=True)

    @app.post("/v1/sub-agent/completions")
    async def sub_agent_completions(req: Request):
        body = await req.json()
        sub = cfg.sub_agent_model
        if sub is None:
            raise HTTPException(status_code=503, detail="No sub_agent model configured")
        body.setdefault("model", sub.id)
        return await _chat_or_completions(req, body, "/v1/completions", chat=False)

    async def _chat_or_completions(req: Request, body: dict, path: str, chat: bool):
        model_id = body.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="`model` is required")
        try:
            mcfg = cfg.by_id(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
        if mcfg.kind not in ("chat", "sub_agent"):
            raise HTTPException(status_code=400, detail=f"Model {model_id!r} is not a chat-capable model")

        # Built-in tool catalog: opt in via body["tools_builtin"] = true.
        # Strip the flag before forwarding so upstream doesn't see it.
        want_builtin_tools = bool(body.pop("tools_builtin", False))
        # Optional cap on tool-call hops to prevent runaway loops.
        max_tool_hops = int(body.pop("max_tool_hops", 4) or 4)
        # Per-request attached documents. The client may pass a structured
        # ``documents`` array; we strip it from the upstream payload and
        # expose it to the document tools (list/read/search) so the model
        # can browse large attachments without inlining all the text.
        request_docs_raw = body.pop("documents", None)
        request_docs: list[dict[str, Any]] = []
        if isinstance(request_docs_raw, list):
            for i, d in enumerate(request_docs_raw):
                if not isinstance(d, dict):
                    continue
                request_docs.append({
                    "id": str(d.get("id") or f"doc{i + 1}"),
                    "name": str(d.get("name") or f"document {i + 1}"),
                    "format": str(d.get("format") or "text"),
                    "text": str(d.get("text") or ""),
                    "size": int(d.get("size") or len(d.get("text") or "")),
                })
        has_docs = bool(request_docs)
        if want_builtin_tools or has_docs:
            body["tools"] = builtin_tools.merge_tools(
                body.get("tools"), want_builtin_tools, has_documents=has_docs,
            )

        # Apply per-model default system prompt when the request has none.
        if chat and getattr(mcfg, "system_prompt", None):
            msgs = body.get("messages") or []
            if not any((m or {}).get("role") == "system" for m in msgs):
                body["messages"] = [{"role": "system", "content": mcfg.system_prompt}, *msgs]

        # When attached documents exist and we exposed the doc tools, inject a
        # short system hint so the model knows what's available without us
        # having to inline the full text.
        if chat and has_docs:
            doc_listing = "\n".join(
                f"- id={d['id']}  name={d['name']!r}  format={d['format']}  length={len(d['text'])}"
                for d in request_docs
            )
            hint = (
                "The user has attached the following documents to this conversation. "
                "Use the tools `list_documents`, `read_document`, and `search_documents` "
                "to inspect them on demand instead of asking the user to repeat their content.\n\n"
                f"{doc_listing}"
            )
            body["messages"] = [{"role": "system", "content": hint}, *(body.get("messages") or [])]

        # RAG augmentation (chat only)
        rag_used = False
        rag_hits_summary: list[dict[str, Any]] = []
        if chat:
            rag_opts = body.pop("rag", None)
            if rag_opts:
                opts = RagOptions(**rag_opts) if not isinstance(rag_opts, RagOptions) else rag_opts
                if opts.enabled:
                    rag_hits_summary = await _augment_with_rag(body, opts)
                    rag_used = bool(rag_hits_summary)

        # Vision pre-processing: when the active chat model has no `mmproj`
        # of its own but the request carries image / audio parts, route those
        # through the small Gemma helper on CUDA0 to extract a text
        # description first, then forward the rewritten messages upstream.
        vision_used: list[dict[str, Any]] = []
        if (
            chat
            and mcfg.kind == "chat"
            and not mcfg.mmproj
            and cfg.vision_model is not None
        ):
            try:
                vision_used = await _preprocess_multimodal(body, model_id)
            except Exception as e:  # noqa: BLE001
                log.warning("vision preprocessing failed: %s", e, exc_info=True)

        try:
            if mcfg.kind == "sub_agent":
                base = await lifecycle.ensure_sub_agent()
            else:
                base = await lifecycle.ensure_chat(model_id)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Failed to start model {model_id}: {e}")

        stream = bool(body.get("stream"))
        target = f"{base}{path}"

        # Activity event metadata
        req_id = uuid.uuid4().hex[:12]
        client_host = req.client.host if req.client else "?"
        ua = req.headers.get("user-agent", "")
        # Pull a short preview from the last user message (chat) or prompt (completions).
        preview = ""
        if chat:
            for m in reversed(body.get("messages") or []):
                if m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, str):
                        preview = c
                    elif isinstance(c, list):
                        preview = " ".join(
                            p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
                        )
                    break
        else:
            p = body.get("prompt")
            preview = p if isinstance(p, str) else (" ".join(p) if isinstance(p, list) else "")
        preview = (preview or "")[:400]

        bus.publish({
            "type": "request.start",
            "id": req_id,
            "path": path,
            "client": client_host,
            "user_agent": ua,
            "model": model_id,
            "stream": stream,
            "rag": rag_used,
            "preview": preview,
            "tools_builtin": want_builtin_tools,
        })
        if rag_hits_summary:
            bus.publish({
                "type": "rag.hits",
                "id": req_id,
                "hits": rag_hits_summary,
            })
        if vision_used:
            bus.publish({
                "type": "vision.used",
                "id": req_id,
                "items": vision_used,
                "via": cfg.vision_model.id if cfg.vision_model else None,
            })

        t0 = time.time()

        # ---- server-side tool-call loop ----
        # If the request advertises any tools (built-in or user-provided), and
        # the upstream model emits matching tool_calls, we execute them and
        # re-call upstream with the tool results appended as messages. This
        # repeats up to `max_tool_hops` times. Once the model produces a final
        # answer (no more tool_calls), we stream/return that response normally.
        # Streaming is preserved for the FINAL hop only; intermediate hops
        # always run with stream=false because we need to inspect the response.
        executed_calls: list[dict[str, Any]] = []
        if chat and body.get("tools") and mcfg.kind != "sub_agent":
            # Make sure upstream sees a proper tool_choice when we have tools.
            body.setdefault("tool_choice", "auto")
            for hop in range(max_tool_hops):
                probe_body = dict(body)
                probe_body["stream"] = False
                try:
                    pr = await proxy_client.post(target, json=probe_body, timeout=httpx.Timeout(60.0, read=None))
                except Exception as e:  # noqa: BLE001
                    raise HTTPException(status_code=502, detail=f"Upstream error during tool hop {hop}: {e}") from e
                try:
                    pj = pr.json()
                except Exception:
                    pj = None
                if pr.status_code >= 400 or not isinstance(pj, dict):
                    # Surface upstream error verbatim.
                    return JSONResponse(status_code=pr.status_code, content=pj or {"error": pr.text})
                choice = (pj.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    # No tool calls were emitted on this hop.
                    if stream:
                        if executed_calls:
                            # Tools already ran on previous hops — the probe
                            # already produced the final answer (with
                            # reasoning + content). Re-issuing with
                            # stream=true now would make the model think
                            # again from scratch and frequently produces an
                            # empty answer. Replay the probe response as SSE
                            # instead so the client sees the actual output.
                            content_text = msg.get("content") or ""
                            reasoning_text = msg.get("reasoning_content") or ""
                            if reasoning_text:
                                bus.publish({"type": "delta", "id": req_id, "reasoning": reasoning_text})
                            if content_text:
                                bus.publish({"type": "delta", "id": req_id, "content": content_text})
                            bus.publish({
                                "type": "request.end", "id": req_id, "ok": True,
                                "status": pr.status_code,
                                "duration_s": time.time() - t0,
                                "tool_calls": len(executed_calls),
                                "usage": pj.get("usage"),
                            })
                            return StreamingResponse(
                                _replay_as_sse(pj, model_id),
                                media_type="text/event-stream",
                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                            )
                        # No tools fired at all → fall through to the regular
                        # streaming branch below so we re-issue with
                        # stream=true and forward live tokens.
                        break
                    bus.publish({
                        "type": "request.end", "id": req_id, "ok": True,
                        "status": pr.status_code,
                        "duration_s": time.time() - t0,
                        "tool_calls": len(executed_calls),
                        "usage": pj.get("usage"),
                    })
                    return JSONResponse(status_code=pr.status_code, content=pj)

                # Append the assistant's tool_calls message verbatim, then
                # execute each call and append the tool result.
                body.setdefault("messages", []).append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                })
                for call in tool_calls:
                    fn = (call.get("function") or {})
                    name = fn.get("name") or ""
                    args_raw = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                    except Exception:
                        args = {}
                    bus.publish({
                        "type": "tool.call", "id": req_id, "name": name,
                        "args_preview": (args_raw if isinstance(args_raw, str) else json.dumps(args))[:300],
                    })
                    if builtin_tools.is_builtin(name):
                        result = await builtin_tools.execute_tool(
                            name, args, http=proxy_client, docs=request_docs,
                        )
                    else:
                        # Unknown tool -> tell the model so it can recover.
                        result = {"error": f"tool {name!r} not available on the server"}
                    executed_calls.append({"name": name, "args": args, "result": result})
                    bus.publish({
                        "type": "tool.result", "id": req_id, "name": name,
                        "result_preview": json.dumps(result)[:400],
                    })
                    body["messages"].append({
                        "role": "tool",
                        "tool_call_id": call.get("id") or name,
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False)[:32000],
                    })
            else:
                # Hop budget exhausted; let the model know and finalize.
                body["messages"].append({
                    "role": "user",
                    "content": "(tool hop limit reached — answer with the information collected so far)",
                })

        if not stream:
            try:
                r = await proxy_client.post(target, json=body)
            except Exception as e:  # noqa: BLE001
                bus.publish({"type": "request.end", "id": req_id, "ok": False, "error": str(e), "duration_s": time.time() - t0})
                raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e
            try:
                payload = r.json()
            except Exception:
                payload = {"error": r.text}
            # Extract assistant text + usage for the activity feed.
            content = ""
            reasoning = ""
            usage = payload.get("usage") if isinstance(payload, dict) else None
            try:
                choice = (payload.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
            except Exception:
                pass
            if content:
                bus.publish({"type": "delta", "id": req_id, "content": content})
            if reasoning:
                bus.publish({"type": "delta", "id": req_id, "reasoning": reasoning})
            bus.publish({
                "type": "request.end",
                "id": req_id,
                "ok": r.status_code < 400,
                "status": r.status_code,
                "duration_s": time.time() - t0,
                "usage": usage,
            })
            return JSONResponse(status_code=r.status_code, content=payload)

        # Streaming: open the upstream stream first so we can forward the real
        # status code / content-type, then relay raw bytes to the client and
        # tee parsed deltas to the event bus.
        upstream_req = proxy_client.build_request("POST", target, json=body)
        try:
            upstream = await proxy_client.send(upstream_req, stream=True)
        except Exception as e:  # noqa: BLE001
            bus.publish({"type": "request.end", "id": req_id, "ok": False, "error": str(e), "duration_s": time.time() - t0})
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}") from e

        async def relay() -> AsyncIterator[bytes]:
            text_buf = ""
            chunks = 0
            usage: dict[str, Any] | None = None
            try:
                async for chunk in upstream.aiter_raw():
                    if not chunk:
                        continue
                    yield chunk
                    # Yield to the event loop so uvicorn flushes this chunk
                    # to the client before we read the next one from upstream.
                    await asyncio.sleep(0)
                    # Tee: decode and parse SSE lines for the activity feed.
                    try:
                        text_buf += chunk.decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    while True:
                        nl = text_buf.find("\n")
                        if nl < 0:
                            break
                        line = text_buf[:nl].rstrip("\r")
                        text_buf = text_buf[nl + 1 :]
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            ev = json.loads(data)
                        except Exception:
                            continue
                        if isinstance(ev, dict) and ev.get("usage"):
                            usage = ev.get("usage")
                        try:
                            delta = (ev.get("choices") or [{}])[0].get("delta") or {}
                        except Exception:
                            delta = {}
                        if delta.get("content"):
                            chunks += 1
                            bus.publish({"type": "delta", "id": req_id, "content": delta["content"]})
                        if delta.get("reasoning_content"):
                            bus.publish({"type": "delta", "id": req_id, "reasoning": delta["reasoning_content"]})
            except Exception as e:  # noqa: BLE001
                bus.publish({"type": "request.end", "id": req_id, "ok": False, "error": str(e), "duration_s": time.time() - t0, "chunks": chunks})
                raise
            else:
                bus.publish({
                    "type": "request.end",
                    "id": req_id,
                    "ok": True,
                    "status": upstream.status_code,
                    "duration_s": time.time() - t0,
                    "chunks": chunks,
                    "usage": usage,
                })
            finally:
                await upstream.aclose()

        media_type = upstream.headers.get("content-type", "text/event-stream")
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-Id": req_id,
        }
        return StreamingResponse(
            relay(),
            status_code=upstream.status_code,
            media_type=media_type,
            headers=headers,
        )

    async def _augment_with_rag(body: dict, opts: RagOptions) -> list[dict[str, Any]]:
        messages = body.get("messages") or []
        query = opts.query
        if not query:
            for m in reversed(messages):
                if m.get("role") == "user":
                    content = m.get("content")
                    if isinstance(content, str):
                        query = content
                    elif isinstance(content, list):
                        # OpenAI multimodal: pick text parts
                        query = " ".join(
                            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                        )
                    break
        if not query:
            return []
        try:
            hits = await rag.query(query, top_k=opts.top_k, source=opts.source, tags=opts.tags)
        except Exception as e:  # noqa: BLE001
            log.warning("RAG query failed; skipping augmentation: %s", e)
            return []
        if not hits:
            return []
        ctx = rag.format_context(hits)
        # Prepend a system message
        new_messages = [{"role": "system", "content": ctx}, *messages]
        body["messages"] = new_messages
        # Build a compact summary for the activity feed.
        summary: list[dict[str, Any]] = []
        for h in hits:
            try:
                meta = getattr(h, "metadata", None) or {}
                doc_id = meta.get("doc_id") or getattr(h, "id", "")
                summary.append({
                    "id": getattr(h, "id", "") or "",
                    "score": float(getattr(h, "score", 0.0) or 0.0),
                    "title": meta.get("title") or doc_id or "",
                    "doc_id": doc_id,
                    "source": meta.get("source") or "",
                    "preview": (getattr(h, "text", "") or "")[:240],
                })
            except Exception:  # noqa: BLE001
                continue
        return summary

    async def _preprocess_multimodal(body: dict, host_model_id: str) -> list[dict[str, Any]]:
        """Caption / transcribe image + audio attachments using the vision
        helper so non-multimodal chat models can still answer about them.

        Walks ``body["messages"]`` in place: every list-shaped ``content``
        with ``image_url`` / ``input_audio`` parts is rewritten so those parts
        become a single text part containing a description produced by the
        vision model. Returns a small summary list (one entry per processed
        attachment) for the activity event feed.
        """
        messages = body.get("messages") or []
        if not messages:
            return []

        # Quick scan to skip work if there are no multimodal parts at all.
        def _has_media(msg: dict[str, Any]) -> bool:
            c = msg.get("content")
            if not isinstance(c, list):
                return False
            for p in c:
                if not isinstance(p, dict):
                    continue
                if p.get("type") in ("image_url", "input_audio", "audio_url"):
                    return True
            return False

        if not any(_has_media(m) for m in messages):
            return []

        vcfg = cfg.vision_model
        if vcfg is None:
            return []
        try:
            vbase = await lifecycle.ensure_vision()
        except Exception as e:  # noqa: BLE001
            log.warning("could not start vision helper: %s", e)
            return []

        used: list[dict[str, Any]] = []
        vision_target = f"{vbase}/v1/chat/completions"

        async def _caption_one(part: dict[str, Any]) -> str:
            kind = part.get("type")
            if kind == "image_url":
                user_parts = [
                    {"type": "text", "text": (
                        "Describe this image in concise detail. Mention any text, "
                        "diagrams, UI elements, people, places and notable colours. "
                        "Reply with the description only."
                    )},
                    part,
                ]
            elif kind in ("input_audio", "audio_url"):
                user_parts = [
                    {"type": "text", "text": (
                        "Transcribe this audio verbatim. If non-speech, briefly "
                        "describe the sound. Reply with the transcript / description only."
                    )},
                    part,
                ]
            else:
                return ""
            payload = {
                "model": vcfg.id,
                "messages": [{"role": "user", "content": user_parts}],
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 16384,
            }
            try:
                r = await proxy_client.post(vision_target, json=payload, timeout=httpx.Timeout(180.0, read=None))
                r.raise_for_status()
                pj = r.json()
            except Exception as e:  # noqa: BLE001
                log.warning("vision caption call failed: %s", e)
                return ""
            try:
                msg = (pj.get("choices") or [{}])[0].get("message") or {}
            except Exception:  # noqa: BLE001
                msg = {}
            # Some llama-server chat templates (notably Gemma + --jinja) route
            # the actual answer into ``reasoning_content`` instead of
            # ``content``. Accept either, preferring the visible content.
            text = (msg.get("content") or "").strip()
            if not text:
                text = (msg.get("reasoning_content") or "").strip()
            if not text:
                # Surface a snippet of the upstream payload so a stuck model
                # (empty output, only stop tokens, etc.) is debuggable.
                try:
                    snippet = json.dumps(pj)[:400]
                except Exception:  # noqa: BLE001
                    snippet = repr(pj)[:400]
                log.warning("vision caption returned empty text; upstream=%s", snippet)
            return text

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new_parts: list[dict[str, Any]] = []
            for p in content:
                if not isinstance(p, dict):
                    new_parts.append(p)
                    continue
                kind = p.get("type")
                if kind in ("image_url", "input_audio", "audio_url"):
                    caption = await _caption_one(p)
                    label = "image" if kind == "image_url" else "audio"
                    if caption:
                        new_parts.append({
                            "type": "text",
                            "text": f"[{label} description from vision helper]\n{caption}",
                        })
                        used.append({
                            "kind": label,
                            "preview": caption[:240],
                            "via": vcfg.id,
                        })
                    else:
                        new_parts.append({
                            "type": "text",
                            "text": f"[{label} attachment could not be processed]",
                        })
                        used.append({"kind": label, "preview": "", "via": vcfg.id, "error": True})
                else:
                    new_parts.append(p)
            msg["content"] = new_parts

        # Refresh idle timer so the watchdog doesn't yank vision out from
        # under a chained request.
        lifecycle.touch_vision()
        return used

    # ---------------- /rag/* ----------------

    @app.post("/rag/ingest")
    async def rag_ingest(payload: IngestRequest) -> dict[str, Any]:
        return await rag.ingest(
            ({"text": d.text, "metadata": d.metadata} for d in payload.documents),
            source=payload.source,
            tags=payload.tags,
        )

    @app.post("/rag/query")
    async def rag_query(payload: QueryRequest) -> dict[str, Any]:
        hits = await rag.query(payload.text, top_k=payload.top_k, source=payload.source, tags=payload.tags)
        return {
            "hits": [
                {"id": h.id, "score": h.score, "text": h.text, "metadata": h.metadata}
                for h in hits
            ]
        }

    @app.get("/rag/stats")
    async def rag_stats() -> dict[str, Any]:
        try:
            return await rag.stats()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"RAG stats failed: {e}")

    @app.get("/rag/documents")
    async def rag_documents(
        source: str | None = None,
        tag: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        try:
            cards = await rag.list_documents(source=source, tag=tag, limit=limit)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"RAG list failed: {e}")
        return {"documents": cards}

    @app.post("/rag/chunks")
    async def rag_chunks(payload: dict[str, Any]) -> dict[str, Any]:
        ids = payload.get("ids") or payload.get("chunk_ids") or []
        if not isinstance(ids, list):
            raise HTTPException(status_code=400, detail="ids must be a list")
        try:
            chunks = await rag.get_chunks([str(i) for i in ids])
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"RAG chunks failed: {e}")
        return {"chunks": chunks}

    @app.delete("/rag/documents")
    async def rag_delete(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await rag.delete(
                chunk_ids=payload.get("chunk_ids"),
                source=payload.get("source"),
                doc_id=payload.get("doc_id"),
                tag=payload.get("tag"),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"RAG delete failed: {e}")

    # ---------------- model swap (admin) ----------------

    @app.post("/admin/load")
    async def admin_load(payload: dict[str, Any]) -> dict[str, Any]:
        model_id = payload.get("model")
        if not model_id:
            raise HTTPException(status_code=400, detail="`model` is required")
        await lifecycle.ensure_chat(model_id)
        return {"active_chat_model": lifecycle.active_chat_model()}

    # ---------------- built-in tools catalog ----------------

    @app.get("/v1/tools")
    async def list_builtin_tools() -> dict[str, Any]:
        return {"tools": builtin_tools.BUILTIN_TOOLS}

    @app.post("/v1/tools/web_search")
    async def call_web_search(payload: dict[str, Any]) -> dict[str, Any]:
        return await builtin_tools.web_search(
            str(payload.get("query", "")),
            int(payload.get("max_results", 5) or 5),
            http=proxy_client,
        )

    @app.post("/v1/tools/web_fetch")
    async def call_web_fetch(payload: dict[str, Any]) -> dict[str, Any]:
        return await builtin_tools.web_fetch(
            str(payload.get("url", "")),
            int(payload.get("max_chars", 8000) or 8000),
            http=proxy_client,
        )

    # ---------------- audio transcription (Whisper) ----------------

    @app.post("/v1/audio/transcriptions")
    async def transcribe(req: Request) -> dict[str, Any]:
        ctype = req.headers.get("content-type", "")
        if "multipart/form-data" not in ctype:
            raise HTTPException(status_code=400, detail="expected multipart/form-data with a `file` field")
        form = await req.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="missing `file` field")
        audio = await upload.read()  # type: ignore[union-attr]
        if not audio:
            raise HTTPException(status_code=400, detail="empty audio")
        model_name = str(form.get("model") or "base")
        language = form.get("language")
        result = await builtin_tools.transcribe_audio(
            audio,
            model_name=model_name,
            language=str(language) if language else None,
        )
        if "error" in result:
            raise HTTPException(status_code=503, detail=result["error"])
        bus.publish({"type": "audio.transcribed", "chars": len(result.get("text", "")), "language": result.get("language")})
        return result

    # ---------------- web UI ----------------

    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def _root_redirect() -> RedirectResponse:
            return RedirectResponse(url="/ui/")
    else:
        log.warning("web UI directory not found: %s", web_dir)

    return app
