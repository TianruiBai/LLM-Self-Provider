"""Lifecycle / process manager for llama-server child processes.

Strategy:
- One persistent embedder child running on `embedding_port`.
- One swappable chat child running on `chat_port`. Switching models kills
  the current child and starts a new one. All swaps are serialized by an
  asyncio.Lock so concurrent /v1/chat/completions for the same model are safe.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

from .registry import ModelConfig, ProviderConfig
from .llama_runner import LlamaRunner

log = logging.getLogger("provider.lifecycle")


def _load_model_override(model_id: str) -> dict:
    """Read ``model_publish`` overrides from the control DB.

    Returns ``{}`` when the DB isn't initialised (e.g. early bootstrap or
    standalone smoke tests). Never raises — overrides are best-effort.
    """
    try:
        from . import db
        row = db.fetchone(
            "SELECT ctx_size, extra_args, system_prompt FROM model_publish "
            "WHERE model_id = ?",
            (model_id,),
        )
    except Exception:  # noqa: BLE001
        return {}
    if row is None:
        return {}
    out: dict = {}
    if row["ctx_size"]:
        out["ctx_size"] = int(row["ctx_size"])
    extra = row["extra_args"]
    if extra:
        try:
            import json
            parsed = json.loads(extra)
            if isinstance(parsed, list):
                out["extra_args"] = [str(x) for x in parsed]
        except Exception:  # noqa: BLE001
            pass
    if row["system_prompt"]:
        out["system_prompt"] = row["system_prompt"]
    return out


class _Child:
    def __init__(self, model: ModelConfig, port: int, proc: subprocess.Popen, log_path: Path):
        self.model = model
        self.port = port
        self.proc = proc
        self.log_path = log_path
        self.started_at = time.time()

    def alive(self) -> bool:
        return self.proc.poll() is None


class LifecycleManager:
    def __init__(self, cfg: ProviderConfig, log_dir: Path | None = None):
        self.cfg = cfg
        self.log_dir = log_dir or Path("logs/provider")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._chat: Optional[_Child] = None
        self._embed: Optional[_Child] = None
        self._sub_agent: Optional[_Child] = None
        self._vision: Optional[_Child] = None
        self._chat_lock = asyncio.Lock()
        self._embed_lock = asyncio.Lock()
        self._sub_agent_lock = asyncio.Lock()
        self._vision_lock = asyncio.Lock()
        # Last-use timestamps for the small-GPU helpers — used by the idle
        # watchdog to free VRAM when nothing has hit them in a while.
        self._embed_last_use: float = 0.0
        self._vision_last_use: float = 0.0
        self._idle_task: Optional[asyncio.Task] = None
        # Hook called when the watchdog auto-unloads a helper, so the gateway
        # can publish an activity event. Set by ``LifecycleManager.set_idle_callback``.
        self._idle_callback = None
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None))
        # Sibling-container llama-server runner for swappable LM Studio GGUFs.
        self._llama = LlamaRunner()
        # Container-backed slots use a flexible naming scheme so we can run
        # one chat container per authenticated user (slot ``chat-<user>``)
        # while keeping the helpers (``embed`` / ``sub_agent`` / ``vision``)
        # as singletons shared across the box. ``_container_active`` is
        # keyed by the full slot string and holds ``{role, model_id}``.
        self._container_active: dict[str, dict[str, str | None]] = {}
        # LRU order of chat user_keys with a live container. Most-recently
        # used at the end. Used to evict when ``_chat_max_concurrent`` is
        # exceeded so per-user containers don't OOM the host.
        self._chat_user_lru: list[str] = []
        try:
            self._chat_max_concurrent = max(
                1, int(os.environ.get("LLAMA_CHAT_MAX_CONCURRENT", "1"))
            )
        except ValueError:
            self._chat_max_concurrent = 1
        # Hard cap on helpers running concurrently on CUDA1 (embed / sub_agent
        # / vision). Acquired by ``acquire_helper`` and released by
        # ``release_helper`` after each /v1/* request finishes, so helpers
        # evict immediately and never pile more than two on the small GPU.
        try:
            helper_cap = max(
                1, int(os.environ.get("LLAMA_HELPER_MAX_CONCURRENT", "2"))
            )
        except ValueError:
            helper_cap = 2
        self._helper_sema = asyncio.Semaphore(helper_cap)
        self._helper_cap = helper_cap

    @staticmethod
    def _user_key(user_key: str | None) -> str:
        """Normalise a user identifier for use as a slot suffix.

        Empty / ``None`` collapses to ``"shared"`` so unauthenticated
        startup paths (RAG warmup, idle watchdog) all share one bucket.
        """
        if not user_key:
            return "shared"
        s = str(user_key).strip().lower()
        # Keep this in sync with llama_runner._sanitize_slot.
        import re as _re
        s = _re.sub(r"[^a-z0-9_-]+", "-", s).strip("-_")
        return (s or "shared")[:32]

    def _chat_slot(self, user_key: str | None) -> str:
        return f"chat-{self._user_key(user_key)}"

    # ---------------- public API ----------------

    async def startup(self) -> None:
        # Embedder, sub-agent and vision helper are spawned lazily on first
        # use to keep the helper GPU (CUDA1) free until something needs it.
        # CUDA0 (RTX PRO 4000) is reserved for the main chat workload. The
        # idle watchdog still tears down stale local-subprocess helpers; the
        # container-backed path uses immediate-after-request eviction
        # instead (see ``release_helper``).
        # Repopulate sibling-container state from any containers left
        # running by a previous gateway process so the UI's "active chat
        # model" pill and Eject button reflect reality immediately.
        await self._reconcile_container_state()
        idle_after = getattr(self.cfg.server, "idle_unload_after_s", 0) or 0
        if idle_after > 0:
            self._idle_task = asyncio.create_task(self._idle_watchdog(idle_after))
        return None

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except (asyncio.CancelledError, Exception):
                pass
            self._idle_task = None
        await self._http.aclose()
        for child in (self._chat, self._embed, self._sub_agent, self._vision):
            if child is not None:
                self._terminate(child)

    def set_idle_callback(self, cb) -> None:
        """Set a callable invoked as ``cb(kind, model_id)`` when the idle
        watchdog auto-unloads a helper child. Used by the gateway to publish
        an activity event."""
        self._idle_callback = cb

    async def ensure_chat(self, model_id: str, user_key: str | None = None) -> str:
        """Ensure the named chat model is the active one for ``user_key``.

        Each authenticated user gets their own sibling container
        (``provider-llama-runner-chat-<user>``) so two users can run
        different chat models concurrently. ``user_key=None`` collapses
        to a shared ``"shared"`` bucket for unauthenticated callers.

        Any non-embedding model is accepted as a chat target: ``chat``,
        ``vision`` (multimodal), and ``sub_agent`` models all expose
        ``/v1/chat/completions`` and can serve user traffic directly.
        """
        cfg = self.cfg.by_id(model_id)
        if cfg.kind == "embedding":
            raise ValueError(f"Model {model_id!r} is an embedding model and cannot serve chat")

        # Sibling-container backends (llama-server, optional vLLM) live as
        # separate Docker services managed by ``LlamaRunner``. We never spawn
        # a local child for them; we just hand the endpoint back.
        if self._is_container_backend(cfg):
            slot = self._chat_slot(user_key)
            await self._enforce_chat_budget(slot)
            ep = await self._resolve_container_endpoint(cfg, slot=slot)
            self._touch_chat_lru(slot)
            # If a llama-server child process is currently active, stop it so
            # VRAM frees up before the sibling container gets traffic.
            async with self._chat_lock:
                if self._chat is not None:
                    log.info("Switching to container chat backend; ejecting %s", self._chat.model.id)
                    self._terminate(self._chat)
                    self._chat = None
            return ep.rstrip("/")

        async with self._chat_lock:
            if self._chat is not None and self._chat.alive() and self._chat.model.id == model_id:
                return self._chat_base_url()

            if self._chat is not None:
                log.info("Swapping chat model: %s -> %s", self._chat.model.id, model_id)
                self._terminate(self._chat)
                self._chat = None

            await self._spawn_chat(cfg)
            return self._chat_base_url()

    async def unload_chat(self, user_key: str | None = None) -> Optional[str]:
        """Stop the active chat child if any. Returns the id that was unloaded.

        ``user_key`` selects which user's per-user container to evict for the
        sibling-container backend; ``None`` means "any chat container" and
        falls back to the most-recently-used one. Local subprocess children
        ignore the user_key (single global slot).
        """
        async with self._chat_lock:
            # Local subprocess child path (legacy).
            if self._chat is not None:
                mid = self._chat.model.id
                log.info("Ejecting chat model %s", mid)
                self._terminate(self._chat)
                self._chat = None
                return mid
            # Container-backed path: pick the slot.
            slots: list[str]
            if user_key is not None:
                slots = [self._chat_slot(user_key)]
            else:
                # Best-effort: try the MRU chat slot first, then any other
                # known chat slots so eject works even when state was lost.
                slots = list(reversed(self._chat_user_lru)) or [
                    s for s in self._container_active.keys() if s.startswith("chat-")
                ]
            for slot in slots:
                active = self._container_active.get(slot) or {}
                role = active.get("role")
                mid = active.get("model_id")
                if mid is not None and role in ("chat", "sub_agent", "vision"):
                    log.info("Ejecting container chat model %s (slot=%s, role=%s)", mid, slot, role)
                    await self._llama.stop(slot)
                    self._container_active.pop(slot, None)
                    if slot in self._chat_user_lru:
                        self._chat_user_lru.remove(slot)
                    return mid
            return None

    async def ensure_sub_agent(self) -> str:
        """Ensure the sub-agent process is running. Returns its base URL."""
        cfg = self.cfg.sub_agent_model
        if cfg is None:
            raise RuntimeError("No sub_agent model configured")
        if self._is_container_backend(cfg):
            # Sub-agent gets its own runner slot pinned to CUDA1 so it can
            # be evicted independently from the main chat container the
            # moment its job finishes (release_helper).
            ep = await self._resolve_container_endpoint(cfg, slot="sub_agent")
            return ep.rstrip("/")
        async with self._sub_agent_lock:
            if self._sub_agent is not None and self._sub_agent.alive():
                return self._sub_agent_base_url()
            await self._spawn_sub_agent(cfg)
            return self._sub_agent_base_url()

    async def ensure_embedder(self) -> str:
        """Ensure the embedder process is running. Returns its base URL."""
        cfg = self.cfg.embedding_model
        if cfg is None:
            raise RuntimeError("No embedding model configured")
        if self._is_container_backend(cfg):
            ep = await self._resolve_container_endpoint(cfg, slot="embed")
            return ep.rstrip("/")
        async with self._embed_lock:
            self._embed_last_use = time.time()
            if self._embed is not None and self._embed.alive():
                return self._embed_base_url()
            await self._spawn_embedder(cfg)
            return self._embed_base_url()

    async def ensure_vision(self) -> str:
        """Ensure the vision (multimodal helper) process is running."""
        cfg = self.cfg.vision_model
        if cfg is None:
            raise RuntimeError("No vision model configured")
        if self._is_container_backend(cfg):
            # Dedicated runner slot on CUDA1 so eviction-after-use doesn't
            # take down a co-tenant chat container.
            ep = await self._resolve_container_endpoint(cfg, slot="vision")
            return ep.rstrip("/")
        async with self._vision_lock:
            self._vision_last_use = time.time()
            if self._vision is not None and self._vision.alive():
                return self._vision_base_url()
            await self._spawn_vision(cfg)
            return self._vision_base_url()

    def touch_vision(self) -> None:
        """Mark the vision helper as recently used (for the idle watchdog)."""
        self._vision_last_use = time.time()

    def touch_embedder(self) -> None:
        self._embed_last_use = time.time()

    @staticmethod
    def _is_container_backend(cfg: ModelConfig) -> bool:
        """Return True when ``cfg`` is served by a sibling Docker container.

        That covers both the modern ``llama-runner://`` sentinel and the
        legacy ``vllm-runner://`` / ``backend=="vllm"`` setups (kept for
        backward compatibility with older ``models.yaml`` files).
        """
        if getattr(cfg, "backend", "llama_cpp") == "vllm":
            return True
        ep = getattr(cfg, "endpoint", None) or ""
        return ep.startswith("llama-runner://") or ep.startswith("vllm-runner://")

    async def _resolve_container_endpoint(self, cfg: ModelConfig, slot: str) -> str:
        """Return the base URL for a sibling-container model.

        If ``cfg.endpoint`` is a ``llama-runner://`` (or legacy
        ``vllm-runner://``) sentinel, ask the runner to ensure a sibling
        llama-server container is up with the requested GGUF; otherwise
        just hand the static endpoint back. Per-model overrides
        (``ctx_size``, ``extra_args``) from the ``model_publish`` table
        are forwarded to the runner.
        """
        ep = cfg.endpoint or ""
        # Legacy ``vllm-runner://`` sentinels are now handled by the
        # llama-server runner — operators that haven't reset their LM Studio
        # registrations get a transparent migration.
        runner_slot = self._llama.parse_endpoint(ep)
        if runner_slot is None and ep.startswith("vllm-runner://"):
            runner_slot = ep[len("vllm-runner://"):].strip("/").split("/")[0] or None
        if runner_slot is not None:
            override = _load_model_override(cfg.id)
            ep_url = await self._llama.ensure(
                slot,
                cfg.path,
                cfg.kind,
                extra_args=override.get("extra_args"),
                ctx_size=override.get("ctx_size"),
                mmproj=getattr(cfg, "mmproj", None),
            )
            # Remember which model id (and role) is parked in this slot so
            # that subsequent ``unload_*`` / ``active_*_model`` calls can
            # find and stop the right sibling container.
            role = (
                "embedding" if cfg.kind == "embedding"
                else "sub_agent" if cfg.kind == "sub_agent"
                else "vision" if cfg.kind == "vision"
                else "chat"
            )
            self._container_active[slot] = {"role": role, "model_id": cfg.id}
            return ep_url
        if not ep:
            raise RuntimeError(f"Container-backed model {cfg.id!r} has no endpoint configured")
        return ep

    async def unload_embedder(self) -> Optional[str]:
        async with self._embed_lock:
            if self._embed is not None:
                mid = self._embed.model.id
                log.info("Ejecting embedder %s", mid)
                self._terminate(self._embed)
                self._embed = None
                return mid
            active = self._container_active.get("embed") or {}
            mid = active.get("model_id")
            if mid is not None:
                log.info("Ejecting container embedder %s", mid)
                await self._llama.stop("embed")
                self._container_active.pop("embed", None)
                return mid
            return None

    async def unload_sub_agent(self) -> Optional[str]:
        async with self._sub_agent_lock:
            if self._sub_agent is not None:
                mid = self._sub_agent.model.id
                log.info("Ejecting sub-agent %s", mid)
                self._terminate(self._sub_agent)
                self._sub_agent = None
                return mid
            active = self._container_active.get("sub_agent") or {}
            mid = active.get("model_id")
            if mid is not None:
                log.info("Ejecting container sub-agent %s", mid)
                await self._llama.stop("sub_agent")
                self._container_active.pop("sub_agent", None)
                return mid
            return None

    async def unload_vision(self) -> Optional[str]:
        async with self._vision_lock:
            if self._vision is not None:
                mid = self._vision.model.id
                log.info("Ejecting vision helper %s", mid)
                self._terminate(self._vision)
                self._vision = None
                return mid
            active = self._container_active.get("vision") or {}
            mid = active.get("model_id")
            if mid is not None:
                log.info("Ejecting container vision helper %s", mid)
                await self._llama.stop("vision")
                self._container_active.pop("vision", None)
                return mid
            return None

    # ---------------- state reconciliation ----------------

    async def _reconcile_container_state(self) -> None:
        """Rehydrate ``_container_active`` + ``_chat_user_lru`` from any
        sibling containers left running by a previous gateway process.

        Without this, after a gateway restart ``active_chat_model()``
        returns ``None`` (so the UI shows "no model loaded" and the Eject
        button is greyed out) even though VRAM is still occupied. We rely
        on the ``provider.slot`` / ``provider.model_path`` labels written
        in :meth:`LlamaRunner._create_and_start`.
        """
        try:
            client = self._llama._docker()
            containers = await asyncio.to_thread(
                client.containers.list, all=False,
                filters={"label": "provider.slot"},
            )
        except Exception:  # noqa: BLE001
            log.debug("container reconciliation skipped (no docker)", exc_info=True)
            return
        for c in containers or []:
            try:
                lbl = c.labels or {}
                slot = lbl.get("provider.slot") or ""
                kind = lbl.get("provider.kind") or ""
                model_path = lbl.get("provider.model_path") or ""
                if not slot or not model_path:
                    continue
                # Map model_path back to a configured model id (best effort).
                model_id = next(
                    (m.id for m in self.cfg.models if m.path == model_path),
                    None,
                )
                if not model_id:
                    continue
                role = (
                    "embedding" if kind == "embedding"
                    else "sub_agent" if kind == "sub_agent"
                    else "vision" if kind == "vision"
                    else "chat"
                )
                self._container_active[slot] = {"role": role, "model_id": model_id}
                if slot.startswith("chat-"):
                    self._touch_chat_lru(slot)
                # Mirror the runner's internal state so subsequent stop()
                # calls find the container.
                from .llama_runner import _SlotState
                self._llama._states[slot] = _SlotState(
                    container_id=c.id,
                    model_path=model_path,
                    endpoint=f"http://{c.name}:{self._llama.port}",
                )
                log.info("Reconciled container slot=%s model=%s", slot, model_id)
            except Exception:  # noqa: BLE001
                log.warning("failed to reconcile container", exc_info=True)

    # ---------------- helper concurrency / immediate-evict ----------------

    async def acquire_helper(self, role: str) -> None:
        """Block until a CUDA1 helper slot is free (max ``LLAMA_HELPER_MAX_CONCURRENT``).

        Pair with :meth:`release_helper` in a try/finally so the semaphore
        always gets returned, even when the upstream request errors out.
        """
        await self._helper_sema.acquire()

    async def release_helper(self, role: str, *, evict: bool = True) -> None:
        """Release the CUDA1 helper slot.

        When ``evict`` is true (default), also tears the helper container
        down so the small GPU is freed for the next task. Tolerates
        repeated calls and missing containers.
        """
        try:
            if evict:
                try:
                    if role == "embedding":
                        await self.unload_embedder()
                    elif role == "sub_agent":
                        await self.unload_sub_agent()
                    elif role == "vision":
                        await self.unload_vision()
                except Exception:  # noqa: BLE001
                    log.warning("release_helper(%s) eviction failed", role, exc_info=True)
        finally:
            try:
                self._helper_sema.release()
            except ValueError:
                # Released more times than acquired — shouldn't happen but
                # don't crash the gateway over it.
                pass

    # ---------------- chat container LRU / budget ----------------

    def _touch_chat_lru(self, slot: str) -> None:
        if slot in self._chat_user_lru:
            self._chat_user_lru.remove(slot)
        self._chat_user_lru.append(slot)

    async def _enforce_chat_budget(self, want_slot: str) -> None:
        """Evict LRU chat containers when the budget would be exceeded.

        ``LLAMA_CHAT_MAX_CONCURRENT`` (default 1) caps how many sibling
        chat containers can run at once across all users so we don't
        accidentally OOM the host when N users connect.
        """
        active = [s for s in self._chat_user_lru if s != want_slot]
        # Reserve one slot for the incoming request.
        while len(active) >= self._chat_max_concurrent:
            victim = active.pop(0)
            log.info("Chat-container budget exceeded; evicting LRU slot %s", victim)
            try:
                await self._llama.stop(victim)
            except Exception:  # noqa: BLE001
                log.warning("LRU eviction of %s failed", victim, exc_info=True)
            self._container_active.pop(victim, None)
            if victim in self._chat_user_lru:
                self._chat_user_lru.remove(victim)

    # ---------------- llama-runner admin pass-throughs ----------------

    async def runner_status(self, slot: str) -> dict:
        """Return current sibling-container state for ``slot`` ("chat"/"embed")."""
        return await self._llama.status(slot)

    async def runner_logs(self, slot: str, n: int = 200) -> str:
        """Return tail of the slot's container logs (or last failure tail)."""
        return await self._llama.tail_logs(slot, n=n)

    async def runner_stop(self, slot: str) -> None:
        """Stop and remove the slot's sibling container; clear failure latch."""
        await self._llama.stop(slot)

    def runner_reset(self, slot: str) -> bool:
        """Clear the sticky failure latch for ``slot`` so retries are allowed."""
        return self._llama.reset_failure(slot)

    def chat_base_url(self) -> Optional[str]:
        if self._chat and self._chat.alive():
            return self._chat_base_url()
        return None

    def embedding_base_url(self) -> Optional[str]:
        if self._embed and self._embed.alive():
            return self._embed_base_url()
        return None

    def sub_agent_base_url(self) -> Optional[str]:
        if self._sub_agent and self._sub_agent.alive():
            return self._sub_agent_base_url()
        return None

    def vision_base_url(self) -> Optional[str]:
        if self._vision and self._vision.alive():
            return self._vision_base_url()
        return None

    def active_chat_model(self) -> Optional[str]:
        if self._chat and self._chat.alive():
            return self._chat.model.id
        # Most-recently-used chat container wins for the global pill. A
        # multi-modal (vision-kind) or sub_agent model selected as the
        # user's chat target also occupies a chat-* slot, so accept those
        # roles here too — otherwise the UI thinks no model is loaded and
        # the Eject button stays disabled.
        for slot in reversed(self._chat_user_lru):
            active = self._container_active.get(slot) or {}
            if active.get("role") in ("chat", "sub_agent", "vision"):
                return active.get("model_id")
        return None

    def active_sub_agent_model(self) -> Optional[str]:
        if self._sub_agent and self._sub_agent.alive():
            return self._sub_agent.model.id
        active = self._container_active.get("sub_agent") or {}
        if active.get("role") == "sub_agent":
            return active.get("model_id")
        return None

    def active_vision_model(self) -> Optional[str]:
        if self._vision and self._vision.alive():
            return self._vision.model.id
        active = self._container_active.get("vision") or {}
        if active.get("role") == "vision":
            return active.get("model_id")
        return None

    # ---------------- internals ----------------

    def _chat_base_url(self) -> str:
        return f"http://{self.cfg.server.host}:{self.cfg.server.chat_port}"

    def _embed_base_url(self) -> str:
        return f"http://{self.cfg.server.host}:{self.cfg.server.embedding_port}"

    def _sub_agent_base_url(self) -> str:
        return f"http://{self.cfg.server.host}:{self.cfg.server.sub_agent_port}"

    def _vision_base_url(self) -> str:
        return f"http://{self.cfg.server.host}:{self.cfg.server.vision_port}"
    async def _spawn_chat(self, model: ModelConfig) -> None:
        port = self.cfg.server.chat_port
        child = self._spawn(model, port)
        self._chat = child
        try:
            await self._wait_ready(port)
        except Exception:
            self._terminate(child)
            self._chat = None
            raise

    async def _spawn_embedder(self, model: ModelConfig) -> None:
        port = self.cfg.server.embedding_port
        child = self._spawn(model, port)
        self._embed = child
        try:
            await self._wait_ready(port)
        except Exception:
            self._terminate(child)
            self._embed = None
            raise

    async def _spawn_sub_agent(self, model: ModelConfig) -> None:
        port = self.cfg.server.sub_agent_port
        child = self._spawn(model, port)
        self._sub_agent = child
        try:
            await self._wait_ready(port)
        except Exception:
            self._terminate(child)
            self._sub_agent = None
            raise

    async def _spawn_vision(self, model: ModelConfig) -> None:
        port = self.cfg.server.vision_port
        child = self._spawn(model, port)
        self._vision = child
        try:
            await self._wait_ready(port)
        except Exception:
            self._terminate(child)
            self._vision = None
            raise

    def _spawn(self, model: ModelConfig, port: int) -> _Child:
        bin_path = model.binary or self.cfg.server.llama_server_bin
        if not Path(bin_path).exists():
            raise FileNotFoundError(f"llama-server not found: {bin_path}")
        if not Path(model.path).exists():
            raise FileNotFoundError(f"model GGUF not found: {model.path}")

        # Apply admin-editable overrides from the control DB (ctx_size,
        # extra_args, system_prompt). Falls back to YAML defaults when the
        # row is missing or columns are NULL.
        override = _load_model_override(model.id)
        args = list(model.args)
        if override.get("ctx_size") and "--ctx-size" not in args and "-c" not in args:
            args += ["--ctx-size", str(int(override["ctx_size"]))]
        if override.get("extra_args"):
            args += list(override["extra_args"])

        # Auto-assign a GPU if the model didn't pin one. Honours explicit
        # ``--device CUDA<n>`` in either YAML args or the admin override.
        from . import gpu as _gpu
        if not _gpu.args_have_device(args):
            role = "chat" if model.kind in ("chat", "sub_agent") else (
                "embed" if model.kind == "embedding" else "vision"
            )
            # sub_agent is a small chat model — bias toward the small GPU
            # so the big chat model keeps its VRAM.
            if model.kind == "sub_agent":
                role = "sub_agent"
            dev = _gpu.pick_device(role)  # type: ignore[arg-type]
            if dev:
                log.info("Auto-assigning %s (%s) to %s", model.id, model.kind, dev)
                args += ["--device", dev]

        cmd = [
            bin_path,
            "--model", model.path,
            "--host", self.cfg.server.host,
            "--port", str(port),
            "--alias", model.id,
            *args,
        ]
        if model.mmproj:
            if not Path(model.mmproj).exists():
                raise FileNotFoundError(f"mmproj not found for {model.id}: {model.mmproj}")
            cmd.extend(["--mmproj", model.mmproj])
        log.info("Spawning llama-server: %s", " ".join(f'"{c}"' if " " in c else c for c in cmd))

        ts = time.strftime("%Y%m%d-%H%M%S")
        safe_id = model.id.replace("/", "_")
        log_path = self.log_dir / f"{safe_id}-{port}-{ts}.log"
        log_f = open(log_path, "w", encoding="utf-8", buffering=1)

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        return _Child(model=model, port=port, proc=proc, log_path=log_path)

    async def _wait_ready(self, port: int) -> None:
        deadline = time.time() + self.cfg.server.startup_timeout_s
        url = f"http://{self.cfg.server.host}:{port}/health"
        last_err: Optional[str] = None
        while time.time() < deadline:
            if port == self.cfg.server.chat_port:
                child = self._chat
            elif port == self.cfg.server.embedding_port:
                child = self._embed
            elif port == self.cfg.server.vision_port:
                child = self._vision
            else:
                child = self._sub_agent
            if child is None or not child.alive():
                code = child.proc.returncode if child else "n/a"
                raise RuntimeError(
                    f"llama-server on port {port} exited early "
                    f"(code={code}); see log: {child.log_path if child else '?'}"
                )
            try:
                r = await self._http.get(url, timeout=5.0)
                if r.status_code == 200:
                    log.info("Server on port %d is ready", port)
                    return
                last_err = f"status={r.status_code}"
            except Exception as e:  # noqa: BLE001
                last_err = repr(e)
            await asyncio.sleep(1.0)
        raise TimeoutError(
            f"llama-server on port {port} not ready within {self.cfg.server.startup_timeout_s}s "
            f"(last_err={last_err})"
        )

    def _terminate(self, child: _Child) -> None:
        if not child.alive():
            return
        log.info("Terminating llama-server pid=%d port=%d (%s)", child.proc.pid, child.port, child.model.id)
        try:
            if os.name == "nt":
                child.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                child.proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        try:
            child.proc.wait(timeout=self.cfg.server.shutdown_timeout_s)
        except subprocess.TimeoutExpired:
            log.warning("Force-killing pid=%d", child.proc.pid)
            child.proc.kill()
            try:
                child.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

    async def _idle_watchdog(self, idle_after: float) -> None:
        """Periodically unload the embedder + vision helper if they have been
        idle for ``idle_after`` seconds. Runs forever until cancelled.

        Both helpers live on the helper GPU (CUDA1 in the dual-GPU setup).
        Container-backed deployments rely on ``release_helper`` for
        immediate post-request eviction; this watchdog only handles the
        local-subprocess fallback.
        """
        try:
            while True:
                await asyncio.sleep(max(15.0, idle_after / 4))
                now = time.time()
                # Embedder
                if self._embed is not None and self._embed.alive() and self._embed_last_use:
                    if now - self._embed_last_use >= idle_after:
                        log.info(
                            "Idle watchdog: unloading embedder (idle for %.0fs)",
                            now - self._embed_last_use,
                        )
                        try:
                            mid = await self.unload_embedder()
                        except Exception:  # noqa: BLE001
                            log.warning("idle unload of embedder failed", exc_info=True)
                            mid = None
                        if mid and self._idle_callback is not None:
                            try:
                                self._idle_callback("embedding", mid)
                            except Exception:  # noqa: BLE001
                                pass
                # Vision
                if self._vision is not None and self._vision.alive() and self._vision_last_use:
                    if now - self._vision_last_use >= idle_after:
                        log.info(
                            "Idle watchdog: unloading vision helper (idle for %.0fs)",
                            now - self._vision_last_use,
                        )
                        try:
                            mid = await self.unload_vision()
                        except Exception:  # noqa: BLE001
                            log.warning("idle unload of vision helper failed", exc_info=True)
                            mid = None
                        if mid and self._idle_callback is not None:
                            try:
                                self._idle_callback("vision", mid)
                            except Exception:  # noqa: BLE001
                                pass
        except asyncio.CancelledError:
            return
