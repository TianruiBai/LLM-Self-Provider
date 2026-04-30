"""Sibling-container vLLM runner (one swappable service per slot).

When a model has ``backend == "vllm"`` and ``endpoint`` of the form
``vllm-runner://<slot>``, the lifecycle layer asks this runner to make sure
a Docker sibling container is up with the right GGUF mounted, then returns
its in-network URL to the gateway.

Design (per user directive: "One swappable vLLM service, restart-on-swap"):

* One sibling container per slot (``chat``, ``embed``).
* Swapping the model in a slot stops the existing container and starts a
  new one with the new ``--model`` argument.
* The gateway communicates with the docker daemon via the bind-mounted
  ``/var/run/docker.sock``.

Environment expected on the gateway container:

  DOCKER_NETWORK         compose network name (default ``provider_default``)
  HF_CACHE_VOLUME        named volume for HF cache (default ``provider_hf-cache``)
  LMSTUDIO_HOST_DIR      host path to mount as ``/lmstudio:ro`` in siblings.
                         Required so the sibling sees the same paths the
                         gateway discovered.
  VLLM_IMAGE             vllm/vllm-openai image tag
                         (default ``vllm/vllm-openai:v0.6.0``)
  VLLM_RUNNER_PORT       internal port (default 8000)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Literal

import httpx

log = logging.getLogger("provider.vllm_runner")

Slot = Literal["chat", "embed"]
RUNNER_SCHEME = "vllm-runner://"

# vLLM CLI flags that take a single value (so we can dedupe by flag name).
_VALUED_FLAGS = {
    "--model", "--host", "--port", "--gpu-memory-utilization",
    "--max-model-len", "--task", "--tokenizer", "--served-model-name",
    "--tensor-parallel-size", "--pipeline-parallel-size", "--dtype",
    "--quantization", "--kv-cache-dtype", "--max-num-seqs",
    "--max-num-batched-tokens", "--swap-space", "--block-size",
    "--seed", "--trust-remote-code", "--download-dir", "--load-format",
    "--config-format", "--enforce-eager", "--rope-scaling", "--rope-theta",
    "--chat-template", "--response-role", "--enable-auto-tool-choice",
    "--tool-call-parser", "--reasoning-parser",
}


def _merge_vllm_args(defaults: list[str], overrides: list[str]) -> list[str]:
    """Merge two argv-style lists; entries in ``overrides`` win for known flags."""
    def _flags(seq: list[str]) -> set[str]:
        return {x for x in seq if x.startswith("-")}

    override_flags = _flags(overrides)
    out: list[str] = []
    i = 0
    while i < len(defaults):
        tok = defaults[i]
        if tok in override_flags:
            # skip default flag and its value (if any)
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


class VllmRunner:
    """Manage swappable vLLM sibling containers via the Docker SDK."""

    def __init__(self) -> None:
        self.network = os.environ.get("DOCKER_NETWORK", "provider_default")
        self.hf_volume = os.environ.get("HF_CACHE_VOLUME", "provider_hf-cache")
        self.lmstudio_host = os.environ.get("LMSTUDIO_HOST_DIR", "")
        # Default to the Blackwell-compatible nightly image. The Qwen3 family
        # (Qwen3.5/3.6) and sm_120 hardware need cu130; the stable ``latest``
        # tag still ships cu128 wheels at the time of writing. Override with
        # ``VLLM_IMAGE`` if you're on Hopper/Ada.
        self.image = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:cu130-nightly")
        self.port = int(os.environ.get("VLLM_RUNNER_PORT", "8000"))
        self._states: dict[str, _SlotState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._client = None  # lazy: docker SDK
        self._http = httpx.AsyncClient(timeout=10.0)

    # ---------- helpers ----------

    def _docker(self):
        if self._client is None:
            import docker  # local import: SDK only required when feature used
            self._client = docker.from_env()
        return self._client

    def _container_name(self, slot: str) -> str:
        return f"provider-vllm-runner-{slot}"

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
        slot: Slot,
        model_path: str,
        kind: str,
        *,
        extra_args: list[str] | None = None,
        ctx_size: int | None = None,
    ) -> str:
        """Make sure slot is running ``model_path``. Returns base URL.

        ``kind`` is the ModelConfig.kind ("chat", "vision", "sub_agent",
        "embedding"). It selects vLLM args (``--task embed`` for embedding).
        ``extra_args`` are appended verbatim to ``vllm serve`` (e.g.
        ``["--tokenizer", "Qwen/Qwen3-...", "--tensor-parallel-size", "1"]``).
        ``ctx_size`` overrides ``--max-model-len`` for chat-like kinds.
        """
        if not self.lmstudio_host:
            raise RuntimeError(
                "LMSTUDIO_HOST_DIR env var not set on gateway; "
                "cannot start sibling vLLM container."
            )
        extra_args = list(extra_args or [])
        # Fingerprint of args so a config change forces a swap.
        cfg_fp = f"{ctx_size or ''}|{'\u0001'.join(extra_args)}"

        async with self._lock(slot):
            state = self._states.setdefault(slot, _SlotState())
            name = self._container_name(slot)

            # Adopt an existing container with the same model_path AND config.
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
                    log.info("vllm-runner: reusing %s for %s", name, model_path)
                    await self._wait_ready(state.endpoint)
                    return state.endpoint
                log.info(
                    "vllm-runner: removing stale %s (had %s, fp=%s)",
                    name, cur_path, cur_fp,
                )
                await asyncio.to_thread(self._remove_container, existing)

            await asyncio.to_thread(
                self._create_and_start,
                name, slot, model_path, kind, extra_args, ctx_size, cfg_fp,
            )
            endpoint = f"http://{name}:{self.port}"
            state.container_id = name
            state.model_path = model_path
            state.endpoint = endpoint
            log.info("vllm-runner: started %s for %s", name, model_path)
            await self._wait_ready(endpoint)
            return endpoint

    async def stop(self, slot: Slot) -> None:
        async with self._lock(slot):
            name = self._container_name(slot)
            existing = await asyncio.to_thread(self._find_container, name)
            if existing is not None:
                await asyncio.to_thread(self._remove_container, existing)
            self._states.pop(slot, None)

    # ---------- docker SDK calls (sync, run via to_thread) ----------

    def _find_container(self, name: str):
        client = self._docker()
        try:
            return client.containers.get(name)
        except Exception:  # NotFound or other
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
        cfg_fp: str,
    ) -> None:
        client = self._docker()
        # Defaults; the user's extra_args win on duplicates.
        defaults: list[str] = [
            "--model", model_path,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--gpu-memory-utilization", os.environ.get("VLLM_GPU_UTIL", "0.85"),
        ]
        if kind == "embedding":
            defaults += ["--task", "embed"]
        else:
            mml = (
                str(ctx_size)
                if ctx_size and ctx_size > 0
                else os.environ.get("VLLM_MAX_MODEL_LEN", "8192")
            )
            defaults += ["--max-model-len", mml]

        cmd = _merge_vllm_args(defaults, extra_args)

        env = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            # Allow --max-model-len above the model's native window so users
            # can opt into YaRN / long-context overrides via extra_args.
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        }
        if os.environ.get("HF_TOKEN"):
            env["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

        # Mounts: HF cache (named volume) + LM Studio dir (host bind, RO).
        volumes = {
            self.hf_volume: {"bind": "/root/.cache/huggingface", "mode": "rw"},
            self.lmstudio_host: {"bind": "/lmstudio", "mode": "ro"},
        }

        # GPU access via NVIDIA container runtime.
        from docker.types import DeviceRequest  # type: ignore
        device_requests = [DeviceRequest(count=-1, capabilities=[["gpu"]])]

        client.containers.run(
            image=self.image,
            command=cmd,
            name=name,
            detach=True,
            environment=env,
            volumes=volumes,
            network=self.network,
            device_requests=device_requests,
            labels={
                "provider.slot": slot,
                "provider.model_path": model_path,
                "provider.kind": kind,
                "provider.cfg_fp": cfg_fp,
            },
            ipc_mode="host",
            shm_size="4g",
            restart_policy={"Name": "unless-stopped"},
        )

    # ---------- readiness ----------

    async def _wait_ready(self, endpoint: str, timeout_s: float = 600.0) -> None:
        deadline = time.time() + timeout_s
        url = f"{endpoint.rstrip('/')}/v1/models"
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                r = await self._http.get(url)
                if r.status_code == 200:
                    return
            except Exception as e:  # noqa: BLE001
                last_err = e
            await asyncio.sleep(2.0)
        raise RuntimeError(
            f"vllm-runner: {endpoint} did not become ready within {timeout_s:.0f}s"
            + (f" (last error: {last_err})" if last_err else "")
        )

    async def aclose(self) -> None:
        try:
            await self._http.aclose()
        except Exception:  # noqa: BLE001
            pass
