"""GPU enumeration and task auto-assignment.

Discovers visible NVIDIA GPUs via ``nvidia-smi`` and exposes a tiny planner
the lifecycle manager uses to pick a device for each model role:

  * **chat**  — biggest available GPU.
  * **embed / sub_agent / vision** — smallest GPU below ``small_threshold_mib``
    (default 9000 MiB ≈ 8 GiB cards). Falls back to the biggest GPU if no
    small device is present (single-card box).

Used at spawn time only when the model's own ``args`` don't already pin a
device with ``--device CUDA<n>``. Existing per-model pins are honoured
verbatim, so handcrafted ``models_local/*/model.yaml`` files keep working.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable, Literal

log = logging.getLogger("provider.gpu")

Role = Literal["chat", "embed", "sub_agent", "vision"]


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    vram_mib: int

    @property
    def device_id(self) -> str:
        return f"CUDA{self.index}"


_CACHE: list[Gpu] | None = None


def list_gpus(refresh: bool = False) -> list[Gpu]:
    """Return GPUs visible to ``nvidia-smi``. Empty list when none found
    (CPU-only host, AMD-only host, or nvidia-smi missing). Cached after the
    first successful call so we don't fork a subprocess per spawn.
    """
    global _CACHE
    if _CACHE is not None and not refresh:
        return _CACHE

    if os.environ.get("PROVIDER_DISABLE_GPU_DISCOVERY") == "1":
        _CACHE = []
        return _CACHE

    smi = shutil.which("nvidia-smi")
    if smi is None:
        log.info("nvidia-smi not on PATH; GPU auto-assignment disabled")
        _CACHE = []
        return _CACHE

    try:
        out = subprocess.check_output(
            [smi, "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("nvidia-smi failed: %s", e)
        _CACHE = []
        return _CACHE

    gpus: list[Gpu] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            mib = int(parts[2])
        except ValueError:
            continue
        gpus.append(Gpu(index=idx, name=parts[1], vram_mib=mib))
    _CACHE = sorted(gpus, key=lambda g: g.index)
    if _CACHE:
        log.info(
            "Discovered %d GPU(s): %s",
            len(_CACHE),
            ", ".join(f"CUDA{g.index} {g.name} ({g.vram_mib} MiB)" for g in _CACHE),
        )
    return _CACHE


def pick_device(role: Role, *, small_threshold_mib: int = 9000) -> str | None:
    """Return ``"CUDA<n>"`` to assign to a model with the given role, or
    ``None`` if no GPU is available (caller should leave llama-server to
    auto-detect or fall back to CPU).

    Dispatch policy (dual-GPU host: CUDA0 = main / RTX PRO 4000,
    CUDA1 = helper):

      * ``chat``                            → CUDA0 (big card, never CUDA1)
      * ``embed`` / ``sub_agent`` / ``vision`` → CUDA1 (helper card)

    The mapping can be overridden per-role via env:

      * ``PROVIDER_CHAT_GPU``    (default ``CUDA0``)
      * ``PROVIDER_HELPER_GPU``  (default ``CUDA1``)

    On single-GPU boxes the only available device wins for every role.
    """
    gpus = list_gpus()
    if not gpus:
        return None
    if len(gpus) == 1:
        return gpus[0].device_id

    chat_pref = os.environ.get("PROVIDER_CHAT_GPU", "CUDA0").strip().upper()
    helper_pref = os.environ.get("PROVIDER_HELPER_GPU", "CUDA1").strip().upper()
    available = {g.device_id for g in gpus}

    if role == "chat":
        # Hard guard: never let a chat / >27B model land on the helper GPU.
        if chat_pref in available and chat_pref != helper_pref:
            return chat_pref
        # Fallback: the biggest card that isn't the helper, or biggest overall.
        non_helper = [g for g in gpus if g.device_id != helper_pref]
        pool = non_helper or gpus
        return max(pool, key=lambda g: g.vram_mib).device_id

    # Helper roles: embed / sub_agent / vision must never share CUDA0
    # with the big chat model when a helper GPU exists.
    if helper_pref in available and helper_pref != chat_pref:
        return helper_pref
    # No dedicated helper GPU configured: fall back to the smallest sub-
    # threshold card, then the smallest card overall.
    small = [g for g in gpus if g.vram_mib < small_threshold_mib]
    if small:
        return min(small, key=lambda g: g.vram_mib).device_id
    return min(gpus, key=lambda g: g.vram_mib).device_id


def args_have_device(args: Iterable[str]) -> bool:
    """True if a model's args already pin ``--device``."""
    for a in args:
        if a == "--device":
            return True
    return False


def topology() -> dict:
    """Plain-dict snapshot for ``GET /admin/gpus``."""
    gpus = list_gpus()
    return {
        "gpus": [
            {"index": g.index, "name": g.name, "vram_mib": g.vram_mib,
             "device": g.device_id}
            for g in gpus
        ],
        "assignment": {
            "chat":      pick_device("chat"),
            "embed":     pick_device("embed"),
            "sub_agent": pick_device("sub_agent"),
            "vision":    pick_device("vision"),
        },
    }
