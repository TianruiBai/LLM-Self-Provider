"""Sibling-container llama.cpp runner (one swappable llama-server per slot).

Mirrors the design of ``provider/vllm_runner.py`` but launches the
``llama.cpp-deepseek-v4-flash`` server image instead of vLLM. This image
includes the antirez DeepSeek-V4-Flash patches alongside upstream model
support, so the same container handles DeepSeek-V4-Flash and ordinary
GGUFs (Qwen3, Llama, Mistral, Gemma, …).

When a model has ``endpoint`` of the form ``llama-runner://<slot>`` the
lifecycle layer asks this runner to make sure a Docker sibling container
is up with the right GGUF mounted, then returns its in-network URL to the
gateway.

Environment expected on the gateway container:

  DOCKER_NETWORK         compose network name (default ``provider_default``)
  HF_CACHE_VOLUME        named volume for HF cache (default ``provider_hf-cache``)
  LMSTUDIO_HOST_DIR      host path mounted as ``/lmstudio:ro`` in siblings.
                         Required so the sibling sees the same paths the
                         gateway discovered.
  LLAMA_IMAGE            llama-server image tag
                         (default ``provider-llama-server:local``)
  LLAMA_RUNNER_PORT      internal port (default 8080 — llama-server default)
  LLAMA_NGL              default ``--n-gpu-layers`` (default ``999``)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass

import httpx

from . import gpu as _gpu

log = logging.getLogger("provider.llama_runner")

# Slot identifiers are now arbitrary strings of the form ``<role>[-<user>]``
# (e.g. ``chat-alice``, ``embed``, ``sub_agent``, ``vision``). The runner
# treats them as opaque keys; lifecycle picks the naming scheme.
RUNNER_SCHEME = "llama-runner://"

_SLOT_SAFE = re.compile(r"[^a-z0-9_-]+")


def _sanitize_slot(slot: str) -> str:
    """Lower-case, strip unsafe chars so the slot is a valid Docker name segment."""
    s = (slot or "").strip().lower()
    s = _SLOT_SAFE.sub("-", s)
    s = s.strip("-_") or "default"
    return s[:48]


_DEVICE_RE = re.compile(r"^cuda(\d+)$", re.IGNORECASE)


def _device_index(spec: str | None) -> int | None:
    """Parse ``"CUDA0"`` / ``"CUDA1"`` / ``"0"`` -> integer index. Returns
    ``None`` for empty / unparseable input so the caller can fall through
    to the count=-1 expose-all behaviour.
    """
    if not spec:
        return None
    s = spec.strip()
    m = _DEVICE_RE.match(s)
    if m:
        return int(m.group(1))
    if s.isdigit():
        return int(s)
    return None


def _strip_device_flag(cmd: list[str]) -> list[str]:
    """Drop any ``--device CUDA<n>`` from a llama-server argv.

    Used when we mask the GPU namespace via Docker's ``device_ids``: inside
    the container the single visible GPU is always index 0, so a stale
    ``--device CUDA1`` from a YAML override would crash the server.
    """
    out: list[str] = []
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok == "--device" and i + 1 < len(cmd):
            i += 2
            continue
        out.append(tok)
        i += 1
    return out

# llama-server CLI flags that take a single value (used to dedupe by flag name).
# Long+short forms listed where common.
_VALUED_FLAGS = {
    "-m", "--model",
    "--host", "--port",
    "-c", "--ctx-size",
    "-ngl", "--n-gpu-layers", "--gpu-layers",
    "--device", "--main-gpu",
    "--split-mode", "--tensor-split",
    "-b", "--batch-size",
    "-ub", "--ubatch-size",
    "-np", "--parallel",
    "-t", "--threads",
    "--alias", "--api-key", "--api-key-file",
    "--mmproj",
    "--rope-freq-base", "--rope-freq-scale", "--rope-scaling",
    "--yarn-orig-ctx", "--yarn-ext-factor", "--yarn-attn-factor",
    "--yarn-beta-slow", "--yarn-beta-fast",
    "--cache-type-k", "--cache-type-v",
    "--seed", "--lora", "--lora-scaled",
    "--chat-template", "--chat-template-file",
    "--reasoning-format",
    "--fit-target",  # DeepSeek-V4-Flash fork extension
}


def _merge_llama_args(defaults: list[str], overrides: list[str]) -> list[str]:
    """Merge two argv-style lists; entries in ``overrides`` win for known flags."""
    def _flags(seq: list[str]) -> set[str]:
        return {x for x in seq if x.startswith("-")}

    override_flags = _flags(overrides)
    out: list[str] = []
    i = 0
    while i < len(defaults):
        tok = defaults[i]
        if tok in override_flags:
            if tok in _VALUED_FLAGS and i + 1 < len(defaults):
                i += 2
            else:
                i += 1
            continue
        out.append(tok)
        i += 1
    out.extend(overrides)
    return out


@dataclass
class _SlotState:
    container_id: str | None = None
    model_path: str | None = None
    endpoint: str | None = None


@dataclass
class _SlotFailure:
    cfg_fp: str
    model_path: str
    error: str
    logs_tail: str
    ts: float


class LlamaRunner:
    """Manage swappable llama-server sibling containers via the Docker SDK."""

    def __init__(self) -> None:
        self.network = os.environ.get("DOCKER_NETWORK", "provider_default")
        self.hf_volume = os.environ.get("HF_CACHE_VOLUME", "provider_hf-cache")
        self.lmstudio_host = os.environ.get("LMSTUDIO_HOST_DIR", "")
        # Default to the locally-built fork image. See
        # ``provider/docs/LLAMA_CPP_CONTAINER.md`` for the build command.
        self.image = os.environ.get("LLAMA_IMAGE", "provider-llama-server:local")
        self.port = int(os.environ.get("LLAMA_RUNNER_PORT", "8080"))
        self.default_ngl = os.environ.get("LLAMA_NGL", "999")
        self._states: dict[str, _SlotState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # GLOBAL container-start serialization. On Windows + WSL2 the GPU
        # paravirtualization driver (``dxgkrnl``) can race when two CUDA
        # containers initialise simultaneously, occasionally taking down
        # Hyper-V with a hard host crash. Serialising container *creation*
        # (not requests against an already-warm server) is enough to avoid
        # the race without harming throughput, since starts are rare.
        self._start_gate = asyncio.Lock()
        # Sticky per-slot failure cache. When a launch fails (bad CLI flag,
        # OOM, missing GGUF, …) we record the (cfg_fp, error, logs) tuple and
        # short-circuit subsequent ``ensure()`` calls for the same fingerprint
        # so that a single misconfiguration does not trigger an infinite
        # crash-respawn loop. Cleared automatically when cfg_fp changes or
        # when an operator calls :meth:`reset_failure`.
        self._failures: dict[str, _SlotFailure] = {}
        self._client = None  # lazy: docker SDK
        self._http = httpx.AsyncClient(timeout=10.0)

    # ---------- helpers ----------

    def _docker(self):
        if self._client is None:
            import docker  # local import: SDK only required when feature used
            self._client = docker.from_env()
        return self._client

    def _container_name(self, slot: str) -> str:
        return f"provider-llama-runner-{_sanitize_slot(slot)}"

    def _lock(self, slot: str) -> asyncio.Lock:
        if slot not in self._locks:
            self._locks[slot] = asyncio.Lock()
        return self._locks[slot]

    def parse_endpoint(self, endpoint: str | None) -> str | None:
        """Return the slot name if endpoint is a runner sentinel, else None."""
        if not endpoint:
            return None
        if not endpoint.startswith(RUNNER_SCHEME):
            return None
        slot = endpoint[len(RUNNER_SCHEME):].strip("/").split("/")[0]
        return slot or None

    # ---------- lifecycle ----------

    async def ensure(
        self,
        slot: str,
        model_path: str,
        kind: str,
        *,
        extra_args: list[str] | None = None,
        ctx_size: int | None = None,
        mmproj: str | None = None,
    ) -> str:
        """Make sure the slot is running ``model_path``. Returns base URL.

        ``kind`` ("chat", "vision", "sub_agent", "embedding") selects defaults
        (e.g. ``--embedding`` for embedding kinds). ``extra_args`` are
        appended verbatim to ``llama-server`` (one token per element).
        ``ctx_size`` overrides ``--ctx-size``. ``mmproj`` adds ``--mmproj``.
        """
        if not self.lmstudio_host:
            raise RuntimeError(
                "LMSTUDIO_HOST_DIR env var not set on gateway; "
                "cannot start sibling llama-server container."
            )
        extra_args = list(extra_args or [])
        cfg_fp = (
            f"{ctx_size or ''}|{mmproj or ''}|{'\u0001'.join(extra_args)}"
        )

        async with self._lock(slot):
            # Short-circuit if we already failed for this exact configuration.
            cached = self._failures.get(slot)
            if (
                cached is not None
                and cached.cfg_fp == cfg_fp
                and cached.model_path == model_path
            ):
                raise RuntimeError(
                    f"llama-runner[{slot}] previous launch failed for "
                    f"{model_path}; refusing to relaunch with same config. "
                    f"Reason: {cached.error}\n"
                    f"--- last container log tail ---\n{cached.logs_tail}\n"
                    f"--- end log tail ---\n"
                    f"Fix the model config or call POST "
                    f"/admin/runner/{slot}/reset to clear this lockout."
                )

            state = self._states.setdefault(slot, _SlotState())
            name = self._container_name(slot)

            existing = await asyncio.to_thread(self._find_container, name)
            if existing is not None:
                lbl = existing.labels or {}
                cur_path = lbl.get("provider.model_path")
                cur_fp = lbl.get("provider.cfg_fp", "")
                if (
                    existing.status == "running"
                    and cur_path == model_path
                    and cur_fp == cfg_fp
                ):
                    state.container_id = existing.id
                    state.model_path = model_path
                    state.endpoint = f"http://{name}:{self.port}"
                    log.info("llama-runner: reusing %s for %s", name, model_path)
                    try:
                        await self._wait_ready(name, state.endpoint)
                    except RuntimeError as e:
                        self._record_failure(slot, cfg_fp, model_path, e)
                        raise
                    return state.endpoint
                log.info(
                    "llama-runner: removing stale %s (had %s, fp=%s)",
                    name, cur_path, cur_fp,
                )
                await asyncio.to_thread(self._remove_container, existing)

            # Serialise actual container creation across slots — see the
            # ``_start_gate`` comment in ``__init__`` for the dxgkrnl race
            # this protects against. Held only while Docker creates the
            # container and the new server initialises CUDA; released
            # before the readiness wait so concurrent traffic isn't blocked.
            async with self._start_gate:
                await asyncio.to_thread(
                    self._create_and_start,
                    name, slot, model_path, kind, extra_args, ctx_size, mmproj, cfg_fp,
                )
                # Brief settle so the NVIDIA driver finishes binding the
                # masked device before the next start runs.
                await asyncio.sleep(2.0)
            endpoint = f"http://{name}:{self.port}"
            state.container_id = name
            state.model_path = model_path
            state.endpoint = endpoint
            log.info("llama-runner: started %s for %s", name, model_path)
            try:
                await self._wait_ready(name, endpoint)
            except RuntimeError as e:
                self._record_failure(slot, cfg_fp, model_path, e)
                raise
            # Successful launch — clear any previous failure for this slot.
            self._failures.pop(slot, None)
            return endpoint

    async def stop(self, slot: str) -> None:
        async with self._lock(slot):
            name = self._container_name(slot)
            existing = await asyncio.to_thread(self._find_container, name)
            if existing is not None:
                await asyncio.to_thread(self._remove_container, existing)
            self._states.pop(slot, None)
            # An explicit stop is also an explicit reset of the failure
            # latch — the operator clearly knows the previous attempt is
            # void and probably plans to retry with different config.
            self._failures.pop(slot, None)

    # ---------- introspection / admin ----------

    def reset_failure(self, slot: str) -> bool:
        """Clear a sticky launch failure for ``slot``. Returns True if cleared."""
        return self._failures.pop(slot, None) is not None

    async def status(self, slot: str) -> dict:
        """Return a dict describing the current container state for ``slot``."""
        name = self._container_name(slot)
        existing = await asyncio.to_thread(self._find_container, name)
        info: dict = {
            "slot": slot,
            "container_name": name,
            "container_status": None,
            "container_exit_code": None,
            "model_path": None,
            "cfg_fp": None,
            "started_at": None,
            "finished_at": None,
            "image": None,
            "failure": None,
        }
        if existing is not None:
            try:
                await asyncio.to_thread(existing.reload)
            except Exception:  # noqa: BLE001
                pass
            attrs = existing.attrs or {}
            state = attrs.get("State", {}) or {}
            lbl = existing.labels or {}
            info.update({
                "container_status": existing.status,
                "container_exit_code": state.get("ExitCode"),
                "model_path": lbl.get("provider.model_path"),
                "cfg_fp": lbl.get("provider.cfg_fp"),
                "started_at": state.get("StartedAt"),
                "finished_at": state.get("FinishedAt"),
                "image": (attrs.get("Config") or {}).get("Image"),
            })
        cached = self._failures.get(slot)
        if cached is not None:
            info["failure"] = {
                "cfg_fp": cached.cfg_fp,
                "model_path": cached.model_path,
                "error": cached.error,
                "logs_tail": cached.logs_tail,
                "ts": cached.ts,
            }
        return info

    async def tail_logs(self, slot: str, n: int = 200) -> str:
        """Return the last ``n`` lines of the slot's container log."""
        name = self._container_name(slot)
        existing = await asyncio.to_thread(self._find_container, name)
        if existing is None:
            cached = self._failures.get(slot)
            if cached is not None:
                return cached.logs_tail
            return ""
        return await asyncio.to_thread(self._fetch_logs, existing, n)

    def _fetch_logs(self, container, n: int = 200) -> str:
        try:
            raw = container.logs(tail=n, stdout=True, stderr=True)
        except Exception as e:  # noqa: BLE001
            return f"<could not read logs: {e}>"
        if isinstance(raw, bytes):
            try:
                return raw.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                return repr(raw)
        return str(raw)

    def _record_failure(
        self,
        slot: str,
        cfg_fp: str,
        model_path: str,
        err: Exception,
    ) -> None:
        name = self._container_name(slot)
        logs_tail = ""
        try:
            existing = self._find_container(name)
            if existing is not None:
                logs_tail = self._fetch_logs(existing, 200)
        except Exception:  # noqa: BLE001
            pass
        self._failures[slot] = _SlotFailure(
            cfg_fp=cfg_fp,
            model_path=model_path,
            error=str(err),
            logs_tail=logs_tail,
            ts=time.time(),
        )

    # ---------- docker SDK calls (sync, run via to_thread) ----------

    def _find_container(self, name: str):
        client = self._docker()
        try:
            return client.containers.get(name)
        except Exception:
            return None

    def _remove_container(self, container) -> None:
        try:
            container.stop(timeout=15)
        except Exception:  # noqa: BLE001
            pass
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass

    def _create_and_start(
        self,
        name: str,
        slot: str,
        model_path: str,
        kind: str,
        extra_args: list[str],
        ctx_size: int | None,
        mmproj: str | None,
        cfg_fp: str,
    ) -> None:
        client = self._docker()
        # llama-server defaults; user's extra_args win on duplicates.
        defaults: list[str] = [
            "--model", model_path,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--n-gpu-layers", self.default_ngl,
            "--alias", os.path.basename(model_path),
        ]
        if kind == "embedding":
            defaults += ["--embedding"]
        # Default context size; let extra_args override.
        ctx = (
            str(ctx_size)
            if ctx_size and ctx_size > 0
            else os.environ.get("LLAMA_CTX_SIZE", "163840")
        )
        defaults += ["--ctx-size", ctx]
        if mmproj:
            defaults += ["--mmproj", mmproj]

        # Optional: disable mmap. On Windows + WSL2, mmap-ing a large GGUF
        # from a Windows-host bind mount can crash the GPU paravirt driver
        # mid-tensor-upload. Setting LLAMA_NO_MMAP=1 forces explicit reads
        # (slower start, much more stable). Default: off — opt in if you
        # see hypervisor wedges loading large models.
        if os.environ.get("LLAMA_NO_MMAP", "").strip().lower() in ("1", "true", "yes", "on"):
            defaults += ["--no-mmap"]

        # Dual-GPU dispatch: pin chat workloads to the big GPU (CUDA0,
        # RTX PRO 4000) and helper workloads (embeddings, sub-agent, vision)
        # to CUDA1, only when the caller hasn't already specified a device.
        if not _gpu.args_have_device(extra_args) and not _gpu.args_have_device(defaults):
            role: _gpu.Role
            if kind == "embedding":
                role = "embed"
            elif kind == "sub_agent":
                role = "sub_agent"
            elif kind == "vision":
                role = "vision"
            else:
                role = "chat"
            dev = _gpu.pick_device(role)
            if dev:
                log.info(
                    "llama-runner[%s]: auto-assigning %s to %s",
                    slot, kind, dev,
                )
                defaults += ["--device", dev]

        cmd = _merge_llama_args(defaults, extra_args)

        # Hard guard: chat / >27B models must never land on the helper GPU
        # (CUDA1 by default). Catches hand-pinned ``--device CUDA1`` in
        # YAML overrides too. Helper roles are allowed on either card so
        # single-GPU boxes still work.
        helper_pref = os.environ.get("PROVIDER_HELPER_GPU", "CUDA1").strip().upper()
        if kind not in ("embedding", "sub_agent", "vision"):
            for i, tok in enumerate(cmd):
                if tok == "--device" and i + 1 < len(cmd):
                    pinned = cmd[i + 1].strip().upper()
                    if pinned == helper_pref:
                        raise RuntimeError(
                            f"llama-runner[{slot}]: refusing to launch "
                            f"chat-kind model {model_path!r} on helper GPU "
                            f"{helper_pref}. CUDA1 is reserved for "
                            f"embeddings / sub_agent / vision. Remove the "
                            f"--device override or set PROVIDER_HELPER_GPU."
                        )
                    break

        env = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            # Pretend to be non-interactive; the fork respects this.
            "LLAMA_ARG_HOST": "0.0.0.0",
        }
        if os.environ.get("HF_TOKEN"):
            env["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

        # Mounts: HF cache (named volume) + LM Studio dir (host bind, RO).
        volumes = {
            self.hf_volume: {"bind": "/root/.cache/huggingface", "mode": "rw"},
            self.lmstudio_host: {"bind": "/lmstudio", "mode": "ro"},
        }

        # GPU access: hard-isolate so llama.cpp can only see ONE card.
        # Without this, ``count=-1`` exposes every GPU to the container and
        # llama.cpp will happily split a 27B/35B model across CUDA0 + CUDA1
        # even when ``--device CUDA0`` is set. Resolve the visible-GPU index
        # from the role (chat -> CUDA0 index, helpers -> CUDA1 index) and
        # drop everything else with a NVIDIA-container-runtime device_id.
        chat_idx = _device_index(os.environ.get("PROVIDER_CHAT_GPU", "CUDA0"))
        helper_idx = _device_index(os.environ.get("PROVIDER_HELPER_GPU", "CUDA1"))
        wanted_idx = helper_idx if kind in ("embedding", "sub_agent", "vision") else chat_idx
        # Honour an explicit per-model ``--device`` if present (admin override).
        for i, tok in enumerate(cmd):
            if tok == "--device" and i + 1 < len(cmd):
                explicit = _device_index(cmd[i + 1])
                if explicit is not None:
                    wanted_idx = explicit
                break
        from docker.types import DeviceRequest  # type: ignore
        if wanted_idx is None:
            device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])]
        else:
            device_requests = [DeviceRequest(
                device_ids=[str(wanted_idx)],
                capabilities=[["gpu"]],
            )]
            # Inside the masked namespace the single visible GPU is index 0.
            # Strip any --device CUDA<n> from the command so llama.cpp doesn't
            # try to address a non-existent CUDA1 inside the container.
            cmd = _strip_device_flag(cmd)
            env["CUDA_VISIBLE_DEVICES"] = "0"

        # Container resource caps. Without these a runaway llama-server
        # (mmap of a 35B GGUF + 160K KV cache spilling to CPU) can consume
        # the entire WSL2 VM on Windows and crash Hyper-V. ``LLAMA_MEM_LIMIT``
        # is the per-container hard cap (e.g. ``"32g"``); empty disables it.
        mem_limit = os.environ.get("LLAMA_MEM_LIMIT", "").strip() or None
        # ``ipc_mode=host`` is only required for multi-GPU NCCL P2P. With our
        # hard 1-GPU isolation it just leaks the WSL2 distro's IPC namespace
        # into every runner and has caused hypervisor instability on Windows.
        # Use a private namespace + a small shm region instead.
        shm_size = os.environ.get("LLAMA_SHM_SIZE", "1g").strip() or "1g"

        run_kwargs = dict(
            image=self.image,
            command=cmd,
            name=name,
            detach=True,
            environment=env,
            volumes=volumes,
            network=self.network,
            device_requests=device_requests,
            labels={
                "provider.slot": _sanitize_slot(slot),
                "provider.model_path": model_path,
                "provider.kind": kind,
                "provider.cfg_fp": cfg_fp,
            },
            shm_size=shm_size,
            # Do NOT auto-restart. Bad CLI flags or model-load failures must
            # surface immediately as an exited container, not a tight crash
            # loop that masks the real error from the gateway.
            restart_policy={"Name": "no"},
        )
        if mem_limit:
            run_kwargs["mem_limit"] = mem_limit
            # Disable swap-out so the container OOMs cleanly inside its own
            # cgroup instead of pressuring the WSL2 page file (which is what
            # actually triggers the hypervisor wedge).
            run_kwargs["memswap_limit"] = mem_limit
        client.containers.run(**run_kwargs)

    # ---------- readiness ----------

    async def _wait_ready(
        self, name: str, endpoint: str, timeout_s: float = 600.0
    ) -> None:
        """Poll ``endpoint``/health until 200, or fail fast if container exits.

        Previously this just polled HTTP for the full timeout window even
        when the container had already exited (e.g. due to a bad CLI flag),
        which produced 10-minute hangs and confusing retry storms. We now
        also reload the container state on each iteration and bail with the
        log tail the moment Docker says the container is no longer running.
        """
        deadline = time.time() + timeout_s
        url = f"{endpoint.rstrip('/')}/health"
        last_err: Exception | None = None
        while time.time() < deadline:
            # 1. Has the container died? If so, stop waiting.
            container = await asyncio.to_thread(self._find_container, name)
            if container is None:
                tail = ""
                cached = self._failures.get(name.rsplit("-", 1)[-1])
                if cached is not None:
                    tail = cached.logs_tail
                raise RuntimeError(
                    f"llama-runner: container {name} disappeared before "
                    f"becoming ready.\n--- log tail ---\n{tail}"
                )
            try:
                await asyncio.to_thread(container.reload)
            except Exception:  # noqa: BLE001
                pass
            status = container.status
            if status in ("exited", "dead", "removing"):
                exit_code = (container.attrs.get("State", {}) or {}).get("ExitCode")
                tail = await asyncio.to_thread(self._fetch_logs, container, 200)
                raise RuntimeError(
                    f"llama-runner: container {name} exited "
                    f"(status={status}, exit_code={exit_code}) before becoming "
                    f"ready.\n--- log tail ---\n{tail}\n--- end log tail ---"
                )

            # 2. Container is up — does the HTTP health endpoint respond?
            try:
                r = await self._http.get(url)
                if r.status_code == 200:
                    return
            except Exception as e:  # noqa: BLE001
                last_err = e
            await asyncio.sleep(2.0)

        # Timed out while still running. Capture logs for the operator.
        container = await asyncio.to_thread(self._find_container, name)
        tail = ""
        if container is not None:
            tail = await asyncio.to_thread(self._fetch_logs, container, 200)
        raise RuntimeError(
            f"llama-runner: {endpoint} did not become ready within {timeout_s:.0f}s"
            + (f" (last http error: {last_err})" if last_err else "")
            + (f"\n--- log tail ---\n{tail}\n--- end log tail ---" if tail else "")
        )

    async def aclose(self) -> None:
        try:
            await self._http.aclose()
        except Exception:  # noqa: BLE001
            pass
