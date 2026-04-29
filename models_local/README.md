# Local model storage

Drop a folder here for each LLM you want to register with the provider service.
On startup, `provider.registry.load_config` scans every direct child folder and
loads any folder containing a `model.yaml` file.

## Folder layout

```
provider/models_local/
  <folder-name>/
    model.yaml         # required — id, kind, args, optional path/mmproj
    *.gguf             # optional — auto-picked as `path` when not specified
    mmproj*.gguf       # optional — auto-picked as `mmproj` (vision projector)
    prompt.md          # optional — auto-loaded as default `system_prompt`
```

## `model.yaml` schema

```yaml
id: "vendor/name"          # unique model id surfaced via /v1/models
kind: chat                 # chat | embedding | sub_agent
path: "C:/abs/path.gguf"   # optional if exactly one *.gguf sits next to model.yaml
binary: "C:/.../llama-server.exe"  # optional per-model llama-server override
mmproj: "C:/.../mmproj.gguf"       # optional vision projector (auto-detected)
system_prompt: |           # optional default system message (auto-loaded from prompt.md)
  You are a helpful assistant.
args:
  - "--ctx-size"
  - "65536"
  - "--device"
  - "CUDA1"
  - "--n-gpu-layers"
  - "999"
```

The whole point of this layout is *drop-in registration*: copy a folder onto
the rig, restart the provider, and the new model shows up in the web UI and
under `/v1/models`. No edits to `models.yaml` required.
