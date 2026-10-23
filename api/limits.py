"""Per-IP fixed-window rate limits, a cap on concurrent heavy computations, and a cap
on request-body size.

In-memory, so they suit one free instance and reset on restart. They stop one
client hogging the service; they are not a defence against a determined attacker."""
import threading
import time
from contextlib import contextmanager

from fastapi.exceptions import HTTPException
from starlette.responses import JSONResponse

TOO_LARGE = "request body too large"


class Busy(Exception):
    """No compute slot was free, and none freed up in time (or too many were already waiting)."""


class RateLimiter:
    def __init__(self, limits, default, window_s=60.0, clock=time.monotonic, max_keys=10_000):
        self._limits, self._default, self._window = dict(limits), default, window_s
        self._clock, self._max_keys = clock, max_keys
        self._lock = threading.Lock()
        self._hits = {}       # (ip, bucket) -> (window start, count)

    def reset(self):
        with self._lock:
            self._hits.clear()

    def _prune(self, now):
        """Drop expired windows; if that is not enough, drop the oldest half. Dropping an
        IP early only hands it a fresh window (fails open); bounded memory matters more
        on a 512 MB instance."""
        self._hits = {k: v for k, v in self._hits.items() if now - v[0] < self._window}
        if len(self._hits) >= self._max_keys:
            keep = sorted(self._hits.items(), key=lambda kv: kv[1][0])[len(self._hits) // 2:]
            self._hits = dict(keep)

    def check(self, ip, path):
        bucket = path if path in self._limits else "*"
        limit = self._limits.get(path, self._default)
        now = self._clock()
        with self._lock:
            if len(self._hits) >= self._max_keys:
                self._prune(now)
            start, count = self._hits.get((ip, bucket), (now, 0))
            if now - start >= self._window:
                start, count = now, 0
            if count >= limit:
                return self._window - (now - start)
            self._hits[(ip, bucket)] = (start, count + 1)
            return None


class ComputeGate:
    """`slots` computations at once. A request that finds them all taken waits up to `wait_s`,
    but only if fewer than `max_waiting` others are already waiting (None: no cap). Each
    waiter ties up a server worker thread, so an uncapped queue could starve the cheap routes."""

    def __init__(self, slots=2, wait_s=15.0, max_waiting=None):
        self._sem = threading.BoundedSemaphore(slots)
        self._wait, self._max_waiting = wait_s, max_waiting
        self._lock = threading.Lock()
        self._waiting = 0

    def _acquire(self):
        if self._sem.acquire(blocking=False):
            return True
        with self._lock:
            if self._max_waiting is not None and self._waiting >= self._max_waiting:
                return False
            self._waiting += 1
        try:
            return self._sem.acquire(timeout=self._wait)
        finally:
            with self._lock:
                self._waiting -= 1

    @contextmanager
    def slot(self):
        if not self._acquire():
            raise Busy()
        try:
            yield
        finally:
            self._sem.release()


class BodyLimit:
    """ASGI middleware: a request body over its path's cap is a 413, before the app parses it.

    A parsed body costs many times its size in memory (a 20 MB body of 1.5M holdings took
    the process from 302 to 799 MiB), so an oversized one must never reach pydantic. A
    declared Content-Length over the cap is refused without reading a byte. A chunked body
    has no declared length, so its bytes are counted as the app pulls them and the request
    fails with a 413 the moment the count passes the cap. Cap lookup is by exact path;
    anything not in `caps` gets `default`."""

    def __init__(self, app, caps, default):
        self.app, self._caps, self._default = app, dict(caps), default

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cap = self._caps.get(scope["path"], self._default)
        declared = _declared_length(scope)
        if declared is not None and declared > cap:
            refusal = JSONResponse(status_code=413, content={"detail": TOO_LARGE})
            return await refusal(scope, receive, send)
        seen = 0

        async def counted():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > cap:
                    # FastAPI re-raises an HTTPException from the body read as itself.
                    raise HTTPException(status_code=413, detail=TOO_LARGE)
            return message

        await self.app(scope, counted, send)


def _declared_length(scope):
    """The Content-Length the client declared, or None if absent or not a number (the
    server rejects a malformed one before the app sees it; the byte count still guards)."""
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def client_ip(request):
    """Render's proxy appends the address it saw; earlier entries are client-supplied."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"
