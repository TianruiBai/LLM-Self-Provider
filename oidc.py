"""OIDC / OAuth2 sign-in (A8).

Pluggable providers configured via ``data/oidc.yaml`` (or ``provider/data/``)
plus environment-variable overrides. Built-in defaults for GitHub and Google;
adding another provider is a matter of dropping its endpoints into the YAML
file. Secrets stay on the server; the browser only ever sees ``state`` +
``code`` round-tripped through the user-agent.

Flow (per RFC 6749 + OIDC + PKCE)::

    1. Browser hits  GET  /auth/oidc/{provider}/start?next=/ui/
    2. Server generates state + code_verifier, stores them in
       ``oauth_states``, redirects the browser to the provider's
       ``authorize_url``.
    3. Provider sends the user back to
       GET /auth/oidc/{provider}/callback?code=...&state=...
    4. Server consumes the state, exchanges the code for an access token
       (with ``code_verifier`` if supported), fetches the user-info
       endpoint, then either logs the user in (existing ``oidc_subject``)
       or creates a brand-new user with ``password=NULL`` and the
       provider:subject pair stamped on it.

This module deliberately avoids ``authlib.integrations.starlette_client``
to keep middleware coupling out of the request lifecycle and make it
straightforward to unit-test (we just patch :func:`_token_exchange` and
:func:`_fetch_userinfo`).
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from . import db

log = logging.getLogger("provider.oidc")

# State tokens expire fairly quickly; a click-through OAuth flow shouldn't
# take more than a couple of minutes.
STATE_TTL_S = 600

# Path resolution mirrors runtime_config.py — try the provider/data dir
# first, then the project-root ``data/`` directory.
_DEFAULT_OIDC_PATHS = (
    Path(__file__).parent / "data" / "oidc.yaml",
    Path(__file__).parent.parent / "data" / "oidc.yaml",
)


# --------------------------------------------------------------- providers


@dataclass
class Provider:
    """An OIDC / OAuth2 identity provider configuration.

    The :attr:`name` is the URL slug used in
    ``/auth/oidc/{name}/start`` etc.
    """

    name: str
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scope: str = "openid email profile"
    # Optional: extra resource fetched alongside userinfo (GitHub keeps the
    # primary email at /user/emails when ``email_url`` is set).
    email_url: Optional[str] = None
    # JSONPath-ish key into the userinfo response that gives the stable
    # subject. ``"id"`` for GitHub, ``"sub"`` for Google.
    subject_key: str = "sub"
    # PKCE: ``True`` for any modern OIDC provider; GitHub does **not**
    # currently support PKCE on the OAuth app flow, so the override is set
    # to False below.
    use_pkce: bool = True
    # Display label for the login UI.
    label: str = ""

    def display_name(self) -> str:
        return self.label or self.name.title()


_BUILTIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "github": {
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
        "email_url": "https://api.github.com/user/emails",
        "scope": "read:user user:email",
        "subject_key": "id",
        "use_pkce": False,
        "label": "GitHub",
    },
    "google": {
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
        "scope": "openid email profile",
        "subject_key": "sub",
        "use_pkce": True,
        "label": "Google",
    },
}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:  # noqa: BLE001
        log.warning("Could not parse OIDC config %s: %s", path, e)
        return {}


def _env(name: str) -> Optional[str]:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else None


# Module-level cache; reload via :func:`reload_providers`.
_PROVIDERS: dict[str, Provider] = {}
_LOADED = False


def load_providers(*, force: bool = False) -> dict[str, Provider]:
    """Resolve provider configs from YAML + env overrides.

    Resolution order for each field, last-wins:

    1. Built-in default (endpoints, scope, subject_key, ...).
    2. ``data/oidc.yaml`` per-provider entry.
    3. Environment variables ``PROVIDER_OIDC_{NAME}_CLIENT_ID`` /
       ``..._CLIENT_SECRET``.

    A provider is *enabled* (and returned) only if both ``client_id`` and
    ``client_secret`` resolve to non-empty strings.
    """
    global _PROVIDERS, _LOADED
    if _LOADED and not force:
        return _PROVIDERS

    raw: dict[str, dict[str, Any]] = {}
    for default_name, defaults in _BUILTIN_DEFAULTS.items():
        raw[default_name] = dict(defaults)

    for path in _DEFAULT_OIDC_PATHS:
        if path.exists():
            file_cfg = _read_yaml(path)
            providers_cfg = file_cfg.get("providers") or {}
            for pname, pcfg in providers_cfg.items():
                if not isinstance(pcfg, dict):
                    continue
                bucket = raw.setdefault(pname, {})
                bucket.update({k: v for k, v in pcfg.items() if v is not None})
            break  # only read the first existing file

    out: dict[str, Provider] = {}
    for pname, cfg in raw.items():
        client_id = cfg.get("client_id") or _env(f"PROVIDER_OIDC_{pname.upper()}_CLIENT_ID")
        client_secret = cfg.get("client_secret") or _env(f"PROVIDER_OIDC_{pname.upper()}_CLIENT_SECRET")
        if not client_id or not client_secret:
            continue
        try:
            out[pname] = Provider(
                name=pname,
                client_id=client_id,
                client_secret=client_secret,
                authorize_url=cfg["authorize_url"],
                token_url=cfg["token_url"],
                userinfo_url=cfg["userinfo_url"],
                scope=cfg.get("scope", "openid email profile"),
                email_url=cfg.get("email_url"),
                subject_key=cfg.get("subject_key", "sub"),
                use_pkce=bool(cfg.get("use_pkce", True)),
                label=cfg.get("label", ""),
            )
        except KeyError as e:
            log.warning("OIDC provider %r missing required field %s", pname, e)

    _PROVIDERS = out
    _LOADED = True
    return out


def reload_providers() -> dict[str, Provider]:
    return load_providers(force=True)


def get_provider(name: str) -> Optional[Provider]:
    return load_providers().get(name)


# --------------------------------------------------------------- state store


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _new_state() -> str:
    return _b64url(secrets.token_bytes(32))


def _new_code_verifier() -> str:
    # PKCE RFC 7636: 43-128 chars, URL-safe.
    return _b64url(secrets.token_bytes(48))


def _challenge_for(verifier: str) -> str:
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def create_state(provider: str, redirect_to: Optional[str], *, with_pkce: bool) -> tuple[str, Optional[str], Optional[str]]:
    """Insert a fresh state row and return ``(state, code_verifier, code_challenge)``.

    ``code_verifier`` and ``code_challenge`` are ``None`` when PKCE is
    disabled for this provider.
    """
    state = _new_state()
    verifier = _new_code_verifier() if with_pkce else None
    challenge = _challenge_for(verifier) if verifier else None
    db.execute(
        "INSERT INTO oauth_states (state, provider, code_verifier, redirect_to, expires_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (state, provider, verifier, redirect_to, db.now_ts() + STATE_TTL_S),
    )
    return state, verifier, challenge


def consume_state(state: str) -> Optional[dict[str, Any]]:
    """Atomically look up and delete a state row. Returns ``None`` if the
    state is unknown, mismatched, or expired."""
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT provider, code_verifier, redirect_to, expires_at FROM oauth_states WHERE state=?",
            (state,),
        ).fetchone()
        conn.execute("DELETE FROM oauth_states WHERE state=?", (state,))
    if row is None:
        return None
    if row["expires_at"] < db.now_ts():
        return None
    return {
        "provider": row["provider"],
        "code_verifier": row["code_verifier"],
        "redirect_to": row["redirect_to"],
    }


def cleanup_expired_states() -> int:
    res = db.execute("DELETE FROM oauth_states WHERE expires_at < ?", (db.now_ts(),))
    try:
        return int(res.rowcount or 0)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return 0


# --------------------------------------------------------------- network


async def _token_exchange(
    provider: Provider,
    code: str,
    redirect_uri: str,
    code_verifier: Optional[str],
) -> dict[str, Any]:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": provider.client_id,
        "client_secret": provider.client_secret,
    }
    if code_verifier:
        data["code_verifier"] = code_verifier
    headers = {"Accept": "application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        r = await client.post(provider.token_url, data=data, headers=headers)
    if r.status_code >= 400:
        raise RuntimeError(f"token exchange failed: {r.status_code} {r.text[:200]}")
    # GitHub returns x-www-form-urlencoded if Accept header is ignored —
    # try JSON first then fall back.
    try:
        return r.json()
    except (json.JSONDecodeError, ValueError):
        from urllib.parse import parse_qs
        return {k: v[0] for k, v in parse_qs(r.text).items()}


async def _fetch_userinfo(provider: Provider, access_token: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "User-Agent": "llm-self-provider",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        r = await client.get(provider.userinfo_url, headers=headers)
        if r.status_code >= 400:
            raise RuntimeError(f"userinfo failed: {r.status_code} {r.text[:200]}")
        info = r.json()
        # GitHub: pull the verified primary email separately if missing.
        if provider.email_url and not info.get("email"):
            try:
                er = await client.get(provider.email_url, headers=headers)
                if er.status_code < 400:
                    emails = er.json() or []
                    primary = next(
                        (e["email"] for e in emails if e.get("primary") and e.get("verified")),
                        None,
                    )
                    if not primary and emails:
                        primary = emails[0].get("email")
                    if primary:
                        info["email"] = primary
            except Exception as e:  # noqa: BLE001
                log.debug("secondary email fetch failed: %s", e)
    return info


# --------------------------------------------------------------- public API


async def exchange_and_fetch(
    provider: Provider,
    code: str,
    redirect_uri: str,
    code_verifier: Optional[str],
) -> dict[str, Any]:
    """Run the OAuth code exchange + fetch the user-info document.

    Returns a dict with at minimum ``subject`` (provider:id) and any of
    ``email``, ``name``, ``login`` that the provider gave us.
    """
    token = await _token_exchange(provider, code, redirect_uri, code_verifier)
    access_token = token.get("access_token")
    if not access_token:
        raise RuntimeError(f"no access_token in response: {token}")
    info = await _fetch_userinfo(provider, access_token)
    raw_subject = info.get(provider.subject_key)
    if raw_subject is None:
        raise RuntimeError(f"userinfo missing {provider.subject_key!r}")
    return {
        "subject": f"{provider.name}:{raw_subject}",
        "email": info.get("email"),
        "name": info.get("name") or info.get("login"),
        "login": info.get("login") or info.get("preferred_username"),
        "raw": info,
    }


def authorize_url_for(
    provider: Provider,
    *,
    state: str,
    redirect_uri: str,
    code_challenge: Optional[str],
) -> str:
    from urllib.parse import urlencode

    params = {
        "client_id": provider.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": provider.scope,
        "state": state,
    }
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"
    sep = "&" if "?" in provider.authorize_url else "?"
    return f"{provider.authorize_url}{sep}{urlencode(params)}"


def derive_username(info: dict[str, Any]) -> str:
    """Pick a unique-ish username out of OIDC user-info."""
    cand = (info.get("login") or info.get("name") or info.get("email") or "").strip()
    if "@" in cand:
        cand = cand.split("@", 1)[0]
    cand = "".join(ch for ch in cand if ch.isalnum() or ch in "._-") or "user"
    return cand[:32]


__all__ = [
    "Provider",
    "STATE_TTL_S",
    "load_providers",
    "reload_providers",
    "get_provider",
    "create_state",
    "consume_state",
    "cleanup_expired_states",
    "exchange_and_fetch",
    "authorize_url_for",
    "derive_username",
]
