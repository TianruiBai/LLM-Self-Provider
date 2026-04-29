"""Hugging Face model downloader with progress reporting.

Used by `/admin/fetch-model`. We prefer `huggingface_hub.hf_hub_download` for
reliable resumable downloads + auth tokens, but fall back to a plain httpx
streaming GET against the public `resolve` endpoint when the library cannot
report progress the way we want (or is unavailable).

Progress is reported through a callback so the gateway can fan it out on the
EventBus.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx

log = logging.getLogger("provider.downloader")

ProgressFn = Callable[[dict], None | Awaitable[None]]


@dataclass
class DownloadResult:
    target_path: Path
    bytes_downloaded: int
    elapsed_s: float
    source: str  # "huggingface_hub" | "https"


class DownloadError(RuntimeError):
    pass


async def _emit(progress: ProgressFn | None, payload: dict) -> None:
    if progress is None:
        return
    res = progress(payload)
    if asyncio.iscoroutine(res):
        await res  # type: ignore[func-returns-value]


async def fetch_hf_file(
    *,
    repo: str,
    filename: str,
    revision: str | None,
    target_dir: Path,
    progress: ProgressFn | None = None,
    final_name: str | None = None,
) -> DownloadResult:
    """Download a single file from a Hugging Face repo into ``target_dir``.

    The result is moved/renamed so the file ends up at
    ``target_dir / (final_name or filename)`` regardless of the resolver
    used. Existing files are returned untouched (idempotent).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    final = target_dir / (final_name or Path(filename).name)
    if final.exists() and final.stat().st_size > 0:
        await _emit(progress, {
            "phase": "skip", "reason": "already-exists",
            "path": str(final), "size": final.stat().st_size,
        })
        return DownloadResult(target_path=final, bytes_downloaded=0, elapsed_s=0.0, source="cache")

    started = time.time()
    # Try the official client first; it handles auth, mirrors, retries and
    # resumable downloads. We get progress by polling the file size.
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except Exception as e:  # noqa: BLE001
        log.info("huggingface_hub unavailable (%s); falling back to direct https", e)
    else:
        try:
            return await _download_via_hf_hub(
                hf_hub_download=hf_hub_download,
                repo=repo,
                filename=filename,
                revision=revision,
                target_dir=target_dir,
                final=final,
                progress=progress,
                started=started,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("huggingface_hub download failed (%s); trying https fallback", e)
            await _emit(progress, {"phase": "fallback", "reason": str(e)})

    return await _download_via_https(
        repo=repo,
        filename=filename,
        revision=revision or "main",
        target_dir=target_dir,
        final=final,
        progress=progress,
        started=started,
    )


async def _download_via_hf_hub(
    *,
    hf_hub_download,
    repo: str,
    filename: str,
    revision: str | None,
    target_dir: Path,
    final: Path,
    progress: ProgressFn | None,
    started: float,
) -> DownloadResult:
    await _emit(progress, {"phase": "start", "source": "huggingface_hub", "repo": repo, "file": filename})

    # Run the (blocking) hf_hub_download in a worker thread while we poll
    # the in-progress filename for size, so the UI sees progress.
    poll_task: asyncio.Task | None = None
    cancel_poll = asyncio.Event()

    async def _poll_once() -> None:
        # The HF cache lays out as <local_dir>/<filename> when local_dir is set.
        in_progress = target_dir / Path(filename).name
        last_seen = -1
        while not cancel_poll.is_set():
            try:
                if in_progress.exists():
                    sz = in_progress.stat().st_size
                    if sz != last_seen:
                        last_seen = sz
                        await _emit(progress, {"phase": "progress", "downloaded": sz})
            except OSError:
                pass
            try:
                await asyncio.wait_for(cancel_poll.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

    poll_task = asyncio.create_task(_poll_once())

    def _do_download() -> str:
        return hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=revision,
            local_dir=str(target_dir),
            # Newer huggingface_hub deprecates symlink controls; this keeps
            # the file as a real copy under our folder.
            local_dir_use_symlinks=False,
        )

    try:
        path_str = await asyncio.to_thread(_do_download)
    finally:
        cancel_poll.set()
        if poll_task is not None:
            try:
                await poll_task
            except Exception:  # noqa: BLE001
                pass

    src = Path(path_str)
    # Move/rename into the canonical location if it differs.
    if src.resolve() != final.resolve():
        if final.exists():
            final.unlink()
        try:
            shutil.move(str(src), str(final))
        except Exception as e:  # noqa: BLE001
            raise DownloadError(f"failed to move {src} -> {final}: {e}") from e

    size = final.stat().st_size
    elapsed = time.time() - started
    await _emit(progress, {"phase": "done", "path": str(final), "size": size, "elapsed_s": elapsed})
    return DownloadResult(target_path=final, bytes_downloaded=size, elapsed_s=elapsed, source="huggingface_hub")


async def _download_via_https(
    *,
    repo: str,
    filename: str,
    revision: str,
    target_dir: Path,
    final: Path,
    progress: ProgressFn | None,
    started: float,
) -> DownloadResult:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    tmp = final.with_suffix(final.suffix + ".part")
    headers = {"User-Agent": "SelfHostedProvider/0.1 (+local)"}
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # Resume support
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    await _emit(progress, {"phase": "start", "source": "https", "repo": repo, "file": filename, "resume_from": resume_from})

    # No total read timeout — large GGUFs take a while.
    timeout = httpx.Timeout(connect=20.0, read=None, write=None, pool=20.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        async with client.stream("GET", url) as r:
            if r.status_code in (401, 403):
                raise DownloadError(
                    f"{r.status_code} from {url}: this repo requires authentication; "
                    "set HUGGING_FACE_HUB_TOKEN or HF_TOKEN in the environment."
                )
            if r.status_code >= 400:
                raise DownloadError(f"{r.status_code} from {url}: {r.text[:200]}")

            total: Optional[int] = None
            cl = r.headers.get("content-length")
            if cl is not None:
                try:
                    total = int(cl) + resume_from
                except ValueError:
                    total = None

            mode = "ab" if resume_from else "wb"
            downloaded = resume_from
            last_emit = 0.0
            with open(tmp, mode) as f:
                async for chunk in r.aiter_bytes(1 << 20):  # 1 MiB chunks
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_emit >= 0.5:
                        last_emit = now
                        await _emit(progress, {
                            "phase": "progress",
                            "downloaded": downloaded,
                            "total": total,
                        })

    # Atomic-ish rename into place.
    if final.exists():
        final.unlink()
    tmp.rename(final)
    size = final.stat().st_size
    elapsed = time.time() - started
    await _emit(progress, {"phase": "done", "path": str(final), "size": size, "elapsed_s": elapsed})
    return DownloadResult(target_path=final, bytes_downloaded=size, elapsed_s=elapsed, source="https")


__all__ = ["fetch_hf_file", "DownloadError", "DownloadResult"]
