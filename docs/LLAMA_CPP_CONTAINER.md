# llama.cpp sibling-container backend

This document explains how the gateway runs GGUF models in production:
**one swappable `llama-server` container per slot**, spawned and torn down
on demand by the gateway.

## Why llama-server (and not vLLM)

We migrated away from the vLLM sibling-container architecture in
April 2026 because:

- Recent Qwen3 / DeepSeek-V4 GGUFs use architectures (`qwen35moe`,
  `deepseek-v4`, MXFP4 MoE quants) that the vLLM nightly's
  `transformers` GGUF reader rejects before our `--hf-config-path`
  override gets a chance to take effect.
- llama.cpp natively understands every quant format that ships in
  `~/.lmstudio/models`, including IQ2_XXS, IQ3_M, MXFP4_MOE and the
  AProj/SExp packed DeepSeek-V4 variants.
- The antirez ``llama.cpp-deepseek-v4-flash`` fork additionally adds
  DeepSeek-V4-Flash inference support — without forking the gateway.

The OpenAI-compatible surface llama-server exposes (`/v1/chat/completions`,
`/v1/embeddings`, `/v1/models`, `/health`) is wire-equivalent to vLLM, so
the gateway's router and audit log are unchanged.

## Build the inference image

The image is built from the fork checked out at
`../llama.cpp-deepseek-v4-flash` (sibling of `provider/`). It uses the
fork's own multi-stage `cuda.Dockerfile` with target `server`.

```bash
# From ./provider
make build-llama
# or, equivalently:
docker compose --profile build-llama build llama-server
```

The result is the image **`provider-llama-server:local`** on your local
Docker daemon. Override the tag by setting `LLAMA_IMAGE` in `.env`.

> First build is heavy — it compiles the fork against CUDA 12.8.1 with
> `GGML_CUDA=ON` and `GGML_BACKEND_DL=ON`. Subsequent rebuilds are
> incremental.

## How the runner works

`provider/llama_runner.py` keeps **one** sibling container per slot
(`chat`, `embed`). When the gateway needs a different model in a slot it:

1. Looks up the existing container by name (`provider-llama-runner-<slot>`).
2. Compares its `provider.model_path` and `provider.cfg_fp` labels
   against what's needed.
3. If the labels match → reuses the container.
4. Otherwise → stops/removes it and starts a fresh one with the new
   `--model` argument and any per-model `extra_args` from the
   `model_publish` table.

The fingerprint covers `ctx_size`, `mmproj`, and `extra_args`, so
admin-published config edits force a swap.

## CLI flag merging

`extra_args` saved in the Account → Models pane are appended verbatim to
`llama-server`. The runner deduplicates known valued flags
(see `_VALUED_FLAGS` in `llama_runner.py`) so user edits override the
gateway's defaults without leaving stale duplicates.

Default args injected by the runner:

```
--model <path>
--host 0.0.0.0
--port 8080
--n-gpu-layers 999            # configurable via LLAMA_NGL
--alias <basename(path)>
--ctx-size <override or 8192>  # configurable via LLAMA_CTX_SIZE
--mmproj <path>                # only when the model has a vision projector
--embedding                    # only when kind == "embedding"
```

## Per-model presets in the web UI

Account → Models → expand a row → Runtime args section now ships
llama-server presets:

| Preset                     | Adds                                                     |
|----------------------------|----------------------------------------------------------|
| Flash attention            | `--flash-attn`                                           |
| Continuous batching        | `--cont-batching --parallel 4`                           |
| Pin to CUDA1               | `--device CUDA1`                                         |
| Dual-GPU layer split       | `--device CUDA1,CUDA0 --split-mode layer`                |
| DeepSeek-V4 stable-164k    | full validated profile (matches `start_deepseek_v4_flash_server.py`) |
| mlock (pin to RAM)         | `--mlock`                                                |

## Environment knobs

| Var                | Default                       | Effect                                            |
|--------------------|-------------------------------|---------------------------------------------------|
| `LLAMA_IMAGE`      | `provider-llama-server:local` | Image used for sibling containers                 |
| `LLAMA_CTX_SIZE`   | `8192`                        | Default `--ctx-size` when no per-model override   |
| `LLAMA_NGL`        | `999`                         | Default `--n-gpu-layers`                          |
| `LLAMA_RUNNER_PORT`| `8080`                        | Internal port the runner listens on               |
| `LMSTUDIO_DIR`     | `~/.lmstudio/models`          | Host path bind-mounted RO at `/lmstudio` in both gateway and sibling |

## Migration from the vLLM runner

Existing LM Studio entries previously registered with
`endpoint="vllm-runner://chat"` continue to work — `lifecycle.py` routes
both `llama-runner://` and the legacy `vllm-runner://` sentinel to the
same llama-server runner. New auto-discoveries register with
`llama-runner://chat`. No DB migration is required.

The old `vllm/vllm-openai` sibling container architecture
(`provider/vllm_runner.py`, the `vllm` compose profile) is kept only as
dead code for repository archaeology. To purge any stale containers from
a previous deployment:

```bash
docker rm -f provider-vllm-runner-chat provider-vllm-runner-embed 2>/dev/null
```

## Troubleshooting

**`docker.errors.ImageNotFound: provider-llama-server:local`** — run
`make build-llama` once on the gateway host.

**`llama-runner: <url> did not become ready within 600s`** — check
sibling logs: `docker logs provider-llama-runner-chat`. Most common
causes are bad `--device` / `--tensor-split` combinations (revert via
the Reset button in the Models pane) and OOM during model load.

**Idle containers eating VRAM** — sibling containers are stopped only
when you switch the slot to a different model or call
`POST /admin/runtime/unload`. There is no idle-watchdog (unlike the
local-binary llama-server children); shut them down manually if needed.
