"""Run with: python -m provider [--config path/to/models.yaml]"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .gateway import create_app
from .registry import load_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="provider", description="Self-hosted model provider service")
    p.add_argument("--config", default=None, help="Path to models.yaml")
    p.add_argument("--host", default=None, help="Override gateway host")
    p.add_argument("--port", type=int, default=None, help="Override gateway port")
    p.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    host = args.host or cfg.gateway.host
    port = args.port or cfg.gateway.port

    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
