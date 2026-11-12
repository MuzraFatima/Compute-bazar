"""HTTP-level performance helpers: cache headers for static assets."""

from __future__ import annotations

from starlette.staticfiles import StaticFiles
from starlette.types import Scope

# Third-party bundles are versioned upstream and never edited here, so browsers
# may keep them for a week. Our own files must revalidate (cheap 304 via ETag).
LONG_LIVED_ASSETS = {"chart.umd.js", "swagger-ui-bundle.js", "swagger-ui.css"}


class CachedStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        if response.status_code in (200, 304):
            name = path.rsplit("/", 1)[-1]
            if name in LONG_LIVED_ASSETS:
                response.headers["Cache-Control"] = "public, max-age=604800"
            else:
                response.headers["Cache-Control"] = "no-cache"
        return response
