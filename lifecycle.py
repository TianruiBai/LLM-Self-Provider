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
from .vllm_runner import VllmRunner

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
        # Sibling-container vLLM runner for swappable LM Studio GGUFs.
        self._vllm = VllmRunner()

    # ---------------- public API ----------------

    async def startup(self) -> None:
        # Embedder, sub-agent and vision helper are spawned lazily on first
        # use to keep the small GPU (CUDA0 / RTX 4070 Laptop) free until
        # something needs it. Start the idle watchdog so they also get torn
        # down again when nothing has used them for a while.
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

    async def ensure_chat(self, model_id: str) -> str:
        """Ensure the named chat model is the active one. Returns base URL.

        Any non-embedding model is accepted as a chat target: ``chat``,
        ``vision`` (multimodal), and ``sub_agent`` models all expose
        ``/v1/chat/completions`` and can serve user traffic directly.
        """
        cfg = self.cfg.by_id(model_id)
        if cfg.kind == "embedding":
            raise ValueError(f"Model {model_id!r} is an embedding model and cannot serve chat")

        # Phase C — vLLM-backed models live as a separate Docker service.
        # We never spawn a child for them; we just hand the endpoint back.
        if getattr(cfg, "backend", "llama_cpp") == "vllm":
            ep = await self._resolve_vllm_endpoint(cfg, slot="chat")
            # If a llama-server child is currently active, stop it so VRAM
            # frees up before vLLM gets traffic.
            async with self._chat_lock:
                if self._chat is not None:
                    log.info("Switching to vLLM chat backend; ejecting %s", self._chat.model.id)
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

    async def unload_chat(self) -> Optional[str]:
        """Stop the active chat child if any. Returns the id that was unloaded."""
        async with self._chat_lock:
            if self._chat is None:
                return None
            mid = self._chat.model.id
            log.info("Ejecting chat model %s", mid)
            self._terminate(self._chat)
            self._chat = None
            return mid

    async def ensure_sub_agent(self) -> str:
        """Ensure the sub-agent process is running. Returns its base URL."""
        cfg = self.cfg.sub_agent_model
        if cfg is None:
            raise RuntimeError("No sub_agent model configured")
        if getattr(cfg, "backend", "llama_cpp") == "vllm":
            ep = await self._resolve_vllm_endpoint(cfg, slot="chat")
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
        if getattr(cfg, "backend", "llama_cpp") == "vllm":
            ep = await self._resolve_vllm_endpoint(cfg, slot="embed")
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
        if getattr(cfg, "backend", "llama_cpp") == "vllm":
            ep = await self._resolve_vllm_endpoint(cfg, slot="chat")
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

    async def _resolve_vllm_endpoint(self, cfg: ModelConfig, slot: str) -> str:
        """Return the base URL for a ``backend=="vllm"`` model.

        If ``cfg.endpoint`` is a ``vllm-runner://`` sentinel, ask the
        sibling-container runner to ensure a vLLM container is up with the
        requested GGUF; otherwise just hand the static endpoint back.
        Per-model overrides (``ctx_size``, ``extra_args``) from the
        ``model_publish`` table are forwarded to the runner.
        """
        ep = cfg.endpoint
        runner_slot = self._vllm.parse_endpoint(ep)
        if runner_slot is not None:
            override = _load_model_override(cfg.id)
            return await self._vllm.ensure(
                slot,
                cfg.path,
                cfg.kind,
                extra_args=override.get("extra_args"),
                ctx_size=override.get("ctx_size"),
            )
        if not ep:
            raise RuntimeError(f"vLLM model {cfg.id!r} has no endpoint configured")
        return ep

    async def unload_embedder(self) -> Optional[str]:
        async with self._embed_lock:
            if self._embed is None:
                return None
            mid = self._embed.model.id
            log.info("Ejecting embedder %s", mid)
            self._terminate(self._embed)
            self._embed = None
            return mid

    async def unload_sub_agent(self) -> Optional[str]:
        async with self._sub_agent_lock:
            if self._sub_agent is None:
                return None
            mid = self._sub_agent.model.id
            log.info("Ejecting sub-agent %s", mid)
            self._terminate(self._sub_agent)
            self._sub_agent = None
            return mid

    async def unload_vision(self) -> Optional[str]:
        async with self._vision_lock:
            if self._vision is None:
                return None
            mid = self._vision.model.id
            log.info("Ejecting vision helper %s", mid)
            self._terminate(self._vision)
            self._vision = None
            return mid

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
        return None

    def active_sub_agent_model(self) -> Optional[str]:
        if self._sub_agent and self._sub_agent.alive():
            return self._sub_agent.model.id
        return None

    def active_vision_model(self) -> Optional[str]:
        if self._vision and self._vision.alive():
            return self._vision.model.id
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

        Both helpers live on the small GPU (CUDA0 by convention), so this is
        what keeps the RTX 4070 Laptop free between bursts of activity.
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
