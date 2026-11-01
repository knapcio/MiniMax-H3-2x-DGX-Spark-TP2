"""Optional per-field form limit for the local video API, off unless explicitly installed."""
from __future__ import annotations

from functools import wraps


MIB = 1024 * 1024
VIDEO_PATHS = frozenset(("/v1/videos", "/v1/videos/sync"))


def install(limit_bytes: int, request_class=None) -> bool:
    """Raise only implicit POST video multipart limits; explicit caller limits win.

    Returns False if an identical patch is already installed. This limits each
    multipart field, not total request size; operators must keep the API private.
    """
    if type(limit_bytes) is not int or not MIB <= limit_bytes <= 64 * MIB:
        raise ValueError("video form limit must be an integer between 1 and 64 MiB")
    if request_class is None:
        from starlette.requests import Request
        request_class = Request
    original = request_class.form
    existing = getattr(original, "_h3_tp2_video_form_limit", None)
    if existing is not None:
        if existing != limit_bytes:
            raise RuntimeError("video form limit is already installed with another value")
        return False

    @wraps(original)
    def form(self, *args, **kwargs):
        scope = getattr(self, "scope", {})
        if (not args and "max_part_size" not in kwargs and scope.get("type") == "http"
                and scope.get("method") == "POST" and scope.get("path") in VIDEO_PATHS):
            kwargs["max_part_size"] = limit_bytes
        return original(self, *args, **kwargs)

    form._h3_tp2_video_form_limit = limit_bytes
    request_class.form = form
    return True
