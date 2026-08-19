"""
Production entrypoint: run Streamlit with pre-compressed static assets.

Streamlit deliberately skips gzip for /static/ (see
streamlit/web/server/starlette/starlette_gzip_middleware.py) because on
localhost the compression CPU costs more than the bandwidth it saves. Over the
public internet the trade-off inverts: this app's first load pulls ~9.3 MB of
uncompressed JS, dominated by the 4.5 MB Plotly bundle.

This module gzips the static bundle once at container-build time and serves the
.gz twins directly, so the win costs zero request-time CPU.

Build step:  python serve.py --precompress
Run:         python serve.py
"""

from __future__ import annotations

import gzip
import mimetypes
import os
import sys
from pathlib import Path

import streamlit

STATIC_ROOT = Path(streamlit.__file__).parent / "static"
COMPRESSIBLE = {".js", ".css", ".html", ".json", ".svg", ".map", ".txt"}
MIN_SIZE = 1024


def precompress(root: Path = STATIC_ROOT) -> None:
    """Write a .gz twin next to every compressible static asset. Idempotent."""
    total_raw = total_gz = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in COMPRESSIBLE:
            continue
        raw = path.stat().st_size
        if raw < MIN_SIZE:
            continue
        target = path.with_name(path.name + ".gz")
        if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
            target.write_bytes(gzip.compress(path.read_bytes(), 9))
        total_raw += raw
        total_gz += target.stat().st_size
    print(f"precompressed {root}: {total_raw/1048576:.1f} MB -> {total_gz/1048576:.1f} MB")


class PrecompressedStatic:
    """Serve <asset>.gz for /static/ requests when the client accepts gzip."""

    def __init__(self, app, root: Path = STATIC_ROOT):
        self.app = app
        self.root = root.resolve()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("GET", "HEAD"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not path.startswith("/static/"):
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        if "gzip" not in headers.get("accept-encoding", ""):
            await self.app(scope, receive, send)
            return

        # Streamlit mounts its static dir at "/", so /static/js/x.js lives at
        # <streamlit>/static/static/js/x.js on disk.
        try:
            asset = (self.root / path.lstrip("/")).resolve()
            asset.relative_to(self.root)  # reject traversal
        except (ValueError, OSError):
            await self.app(scope, receive, send)
            return

        gz = asset.with_name(asset.name + ".gz")
        if not gz.is_file():
            await self.app(scope, receive, send)
            return

        from starlette.responses import FileResponse

        media_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
        response = FileResponse(
            gz,
            media_type=media_type,
            headers={
                "content-encoding": "gzip",
                "cache-control": "public, immutable, max-age=31536000",
                "vary": "Accept-Encoding",
            },
        )
        await response(scope, receive, send)


def _install() -> None:
    """Append PrecompressedStatic to Streamlit's middleware stack.

    This reaches into Streamlit internals, so a version bump can move the hook.
    Losing compression is a performance regression, not an outage — degrade to a
    warning rather than failing to boot.
    """
    try:
        from starlette.middleware import Middleware
        from streamlit.web.server.starlette import starlette_app

        original = starlette_app.create_streamlit_middleware

        def patched():
            return [Middleware(PrecompressedStatic), *original()]

        starlette_app.create_streamlit_middleware = patched
    except (ImportError, AttributeError) as exc:
        print(
            f"WARNING: serving static assets uncompressed — {exc}. "
            "Streamlit's middleware hook moved; update serve.py._install().",
            file=sys.stderr,
        )


def main() -> None:
    if "--precompress" in sys.argv:
        precompress()
        return

    _install()

    # Delegate to Streamlit's own CLI so every --server.* flag keeps working.
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    sys.argv = ["streamlit", "run", app, *sys.argv[1:]]

    from streamlit.web.cli import main as streamlit_main

    streamlit_main()


if __name__ == "__main__":
    main()
