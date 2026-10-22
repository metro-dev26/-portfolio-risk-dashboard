"""Per-IP fixed-window rate limits and a cap on concurrent heavy computations.

In-memory, so they suit one free instance and reset on restart. They stop one
client hogging the service; they are not a defence against a determined attacker."""
import threading
import time
from contextlib import contextmanager


class Busy(Exception):
    """Every compute slot stayed taken for the whole wait."""


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
    def __init__(self, slots=2, wait_s=15.0):
        self._sem, self._wait = threading.BoundedSemaphore(slots), wait_s

    @contextmanager
    def slot(self):
        if not self._sem.acquire(timeout=self._wait):
            raise Busy()
        try:
            yield
        finally:
            self._sem.release()


def client_ip(request):
    """Render's proxy appends the address it saw; earlier entries are client-supplied."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"
