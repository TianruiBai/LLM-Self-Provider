"""Load and validate models.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


ModelKind = Literal["chat", "embedding", "sub_agent", "vision"]


@dataclass
class ModelConfig:
    id: str
    kind: ModelKind
    path: str
    args: list[str] = field(default_factory=list)
    # Optional per-model llama-server binary override. DeepSeek-V4-Flash
    # requires a forked CUDA build that lives next to the source tree, while
    # everything else uses the shared Vulkan binary from `server.llama_server_bin`.
    binary: str | None = None
    # Optional vision projector ("image matrix") for multimodal models.
    # llama-server picks this up via --mmproj.
    mmproj: str | None = None
    # Default system prompt prepended to chat completions if the request
    # carries no system message. Loaded from `prompt.md` when discovered
    # via `models_dir`.
    system_prompt: str | None = None
    # Folder this model was loaded from (when discovered via models_dir).
    # Useful for downstream tools that want to read sibling resources.
    folder: str | None = None
    # Optional auto-download spec. Used by /admin/fetch-model to pull the
    # GGUF (and optionally an mmproj) from Hugging Face into `folder`.
    #   download:
    #     repo: "unsloth/Qwen3.5-122B-A10B-GGUF"
    #     file: "Qwen3.5-122B-A10B-MXFP4.gguf"
    #     mmproj_file: "mmproj-F16.gguf"   # optional vision projector
    #     revision: "main"                  # optional
    download: dict | None = None
    # Phase C — backend selector. ``llama_cpp`` (default) spawns a local
    # llama-server child; ``vllm`` treats ``endpoint`` as an OpenAI-
    # compatible upstream (typically a Docker compose service).
    backend: Literal["llama_cpp", "vllm"] = "llama_cpp"
    # Required when ``backend == "vllm"``. Example: "http://vllm-chat:8000".
    endpoint: str | None = None


@dataclass
class ServerConfig:
    llama_server_bin: str = ""
    host: str = "127.0.0.1"
    chat_port: int = 18001
    embedding_port: int = 18002
    sub_agent_port: int = 18003
    vision_port: int = 18004
    startup_timeout_s: int = 600
    shutdown_timeout_s: int = 30
    # Auto-unload helpers (embedder + vision) pinned to the small GPU after
    # this many seconds of inactivity, so the RTX 4070 Laptop stays free for
    # other workloads. Set to 0 to disable.
    idle_unload_after_s: int = 300
    # Optional folder to auto-discover model definitions. Each subdirectory
    # may contain a `model.yaml` (full ModelConfig) and a `*.gguf` weight,
    # an optional `mmproj*.gguf` (vision projector) and a `prompt.md`.
    models_dir: str | None = None
    # Optional path to an LM Studio model cache (defaults to ``~/.lmstudio/
    # models`` when present). Layout assumed:
    #   <root>/<publisher>/<model_name>/*.gguf
    # Each .gguf weight discovered there is registered as a ``chat`` model
    # with id ``lmstudio/<publisher>/<model_name>``. Admins decide which of
    # them are published to end users via the ``model_publish`` table.
    lmstudio_dir: str | None = None


@dataclass
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class RagConfig:
    backend: str = "mongo"  # "mongo" | "lance"
    mongo_uri: str = "mongodb://127.0.0.1:27017/?directConnection=true"
    database: str = "provider_rag"
    collection: str = "documents"
    vector_index: str = "vector_index"
    embedding_dim: int = 4096
    default_top_k: int = 4
    chunk_chars: int = 1200
    chunk_overlap: int = 150
    lance_dir: str = "data/lance"


@dataclass
class ProviderConfig:
    server: ServerConfig
    gateway: GatewayConfig
    rag: RagConfig
    models: list[ModelConfig]

    def by_id(self, model_id: str) -> ModelConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"Unknown model id: {model_id!r}")

    @property
    def chat_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.kind == "chat"]

    @property
    def embedding_model(self) -> ModelConfig | None:
        for m in self.models:
            if m.kind == "embedding":
                return m
        return None

    @property
    def sub_agent_model(self) -> ModelConfig | None:
        for m in self.models:
            if m.kind == "sub_agent":
                return m
        return None

    @property
    def vision_model(self) -> ModelConfig | None:
        for m in self.models:
            if m.kind == "vision":
                return m
        return None


def load_config(path: str | Path | None = None) -> ProviderConfig:
    if path is None:
        path = Path(__file__).parent / "models.yaml"
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    server = ServerConfig(**raw["server"])
    gateway = GatewayConfig(**raw.get("gateway", {}))
    rag = RagConfig(**raw.get("rag", {}))
    models = [ModelConfig(**m) for m in raw["models"]]

    # Auto-discover models from the local storage folder (drop-in registration).
    if server.models_dir:
        base_dir = Path(server.models_dir)
        if not base_dir.is_absolute():
            base_dir = path.parent / base_dir
        if base_dir.is_dir():
            for entry in sorted(base_dir.iterdir()):
                if not entry.is_dir():
                    continue
                discovered = _load_model_folder(entry)
                if discovered is not None:
                    models.append(discovered)
        else:
            print(f"[registry] models_dir does not exist: {base_dir}")

    # Auto-discover GGUFs under the user's LM Studio cache. The default
    # location is ``~/.lmstudio/models``; admins can point ``lmstudio_dir``
    # elsewhere. Models are *registered* but stay invisible to non-admin
    # users until an admin explicitly publishes them via /admin/models.
    seen_ids = {m.id for m in models}
    for discovered in _discover_lmstudio(server.lmstudio_dir):
        if discovered.id in seen_ids:
            continue
        models.append(discovered)
        seen_ids.add(discovered.id)

    seen: set[str] = set()
    for m in models:
        if m.id in seen:
            raise ValueError(f"Duplicate model id in registry: {m.id}")
        seen.add(m.id)
        if m.backend == "vllm":
            # vLLM models live in a remote container; no local file to check.
            continue
        if not Path(m.path).exists():
            # Don't fail hard — warn at startup, fail at load time.
            print(f"[registry] WARNING: model file not found: {m.path}")
        if m.mmproj and not Path(m.mmproj).exists():
            print(f"[registry] WARNING: mmproj not found for {m.id}: {m.mmproj}")
        if m.kind not in ("chat", "embedding", "sub_agent", "vision"):
            raise ValueError(f"Invalid kind for {m.id}: {m.kind}")

    if server.llama_server_bin and not Path(server.llama_server_bin).exists():
        print(f"[registry] WARNING: llama-server not found: {server.llama_server_bin}")

    return ProviderConfig(server=server, gateway=gateway, rag=rag, models=models)


def _load_model_folder(folder: Path) -> ModelConfig | None:
    """Discover a model definition from a local storage folder.

    Convention:
      <folder>/model.yaml      -> ModelConfig fields (id, kind, args, ...)
      <folder>/*.gguf          -> picked as `path` if not given (excludes mmproj*.gguf)
      <folder>/mmproj*.gguf    -> picked as `mmproj` if not given
      <folder>/prompt.md       -> picked as `system_prompt` if not given
    """
    yaml_path = folder / "model.yaml"
    if not yaml_path.exists():
        # Silently skip folders that are not model definitions (e.g. README only).
        return None
    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        print(f"[registry] failed to parse {yaml_path}: {e}")
        return None
    if not isinstance(data, dict):
        print(f"[registry] {yaml_path} must be a mapping")
        return None

    # Auto-detect weight file.
    if not data.get("path"):
        weights = sorted(
            p for p in folder.glob("*.gguf")
            if not p.name.lower().startswith(("mmproj", "mm-proj", "vision"))
        )
        if weights:
            data["path"] = str(weights[0])
    # Auto-detect mmproj.
    if not data.get("mmproj"):
        proj = sorted(folder.glob("mmproj*.gguf")) + sorted(folder.glob("mm-proj*.gguf"))
        if proj:
            data["mmproj"] = str(proj[0])
    # Auto-load default system prompt.
    if not data.get("system_prompt"):
        prompt = folder / "prompt.md"
        if prompt.exists():
            try:
                data["system_prompt"] = prompt.read_text(encoding="utf-8").strip()
            except Exception as e:  # noqa: BLE001
                print(f"[registry] failed to read {prompt}: {e}")
    data["folder"] = str(folder)

    if not data.get("id") or not data.get("kind") or not data.get("path"):
        print(f"[registry] {yaml_path} missing id/kind/path; skipping")
        return None
    # Drop unknown keys to keep ModelConfig forward-compatible.
    allowed = {"id", "kind", "path", "args", "binary", "mmproj", "system_prompt",
               "folder", "download", "backend", "endpoint"}
    data = {k: v for k, v in data.items() if k in allowed}
    return ModelConfig(**data)


# ---------------------------------------------------------------- LM Studio

def _default_lmstudio_dir() -> Path | None:
    """Return ``~/.lmstudio/models`` if it exists on this machine."""
    candidates = [
        Path.home() / ".lmstudio" / "models",
        Path.home() / ".cache" / "lm-studio" / "models",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _discover_lmstudio(explicit: str | None) -> list[ModelConfig]:
    """Walk an LM Studio model cache and register each GGUF as a chat model.

    Layout (LM Studio convention):
        <root>/<publisher>/<model_name>/<weights>.gguf
        <root>/<publisher>/<model_name>/mmproj-*.gguf   (optional)

    The first non-``mmproj`` ``.gguf`` per folder becomes the model weight.
    Resulting id: ``lmstudio/<publisher>/<model_name>``. Multimodal repos
    that ship an ``mmproj-*.gguf`` are auto-tagged ``kind=vision``.
    """
    root = Path(explicit) if explicit else _default_lmstudio_dir()
    if root is None or not root.is_dir():
        return []

    out: list[ModelConfig] = []
    for publisher in sorted(p for p in root.iterdir() if p.is_dir()):
        for model_dir in sorted(m for m in publisher.iterdir() if m.is_dir()):
            weights = sorted(
                p for p in model_dir.glob("*.gguf")
                if not p.name.lower().startswith(("mmproj", "mm-proj"))
            )
            if not weights:
                continue
            mmproj = sorted(model_dir.glob("mmproj*.gguf")) + sorted(
                model_dir.glob("mm-proj*.gguf")
            )
            kind: ModelKind = "vision" if mmproj else "chat"
            mid = f"lmstudio/{publisher.name}/{model_dir.name}"
            out.append(
                ModelConfig(
                    id=mid,
                    kind=kind,
                    path=str(weights[0]),
                    mmproj=str(mmproj[0]) if mmproj else None,
                    folder=str(model_dir),
                    # In-container deployment: GGUFs are run by a swappable
                    # sibling llama-server container managed by
                    # ``provider.llama_runner``. The sentinel endpoint tells
                    # lifecycle to invoke the runner rather than treat it as
                    # a static URL.
                    backend="llama_cpp",
                    endpoint="llama-runner://chat",
                )
            )
    if out:
        print(f"[registry] discovered {len(out)} LM Studio model(s) under {root}")
    return out
