"""Async TVCache client over one persistent connection.

Each rollout owns one TVCacheClient (and therefore one connection); requests
on a connection are sequential, and concurrency comes from many rollouts each
holding their own client. Coroutines may share a client: a lock serializes
their requests on the connection.
"""

import asyncio
import json
import logging
import time
from typing import List, Optional

from tvclient import protocol
from tvclient.protocol import (
    ClaimOutcome,
    ClaimResponse,
    FindResponse,
    ForkCost,
    InsertResult,
    JsonResponse,
    MarkResult,
    ReclaimableSandbox,
    ReclaimableSandboxes,
    SandboxCandidate,
    SandboxCandidates,
    SandboxEvictionRecord,
    TVCacheServerError,
    UnrefResult,
)


class FindResult:
    """A find response plus the lease on the returned sandbox.

    The server increments the sandbox's reference count whenever find returns a
    sandbox_id, so every FindResult must be released exactly once. Use it as an
    async context manager:

        async with await client.find(task_id, path) as found:
            ...  # fork here; the snapshot cannot be reclaimed while leased

    release() is idempotent and a no-op when there is no sandbox, and it runs
    even when the body raises, so a crashed rollout cannot pin an eviction
    victim forever.
    """

    def __init__(self, client: "TVCacheClient", task_id: str, response: FindResponse):
        self.value = response.value
        self.sandbox_path = response.sandbox_path
        self.sandbox_id = response.sandbox_id
        self._client = client
        self._task_id = task_id
        self._released = response.sandbox_id is None

    async def release(self) -> None:
        """Decrements the sandbox's reference count on the server (at most once)."""
        if self._released:
            return
        self._released = True
        await self._client.unref_sandbox(self._task_id, self.sandbox_id)

    async def __aenter__(self) -> "FindResult":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.release()


class ClaimResult(FindResult):
    """A claim response plus the lease on the prefix sandbox, if one was returned.

    The server leases a sandbox only when some entry must be executed (owner
    or missing), so release it exactly like a FindResult:

        async with await client.claim(task_id, prefix, entries) as claimed:
            ...  # claimed.outcomes[i] is the outcome for entries[i]
    """

    def __init__(self, client: "TVCacheClient", task_id: str, response: ClaimResponse):
        super().__init__(
            client, task_id, FindResponse(None, response.sandbox_path, response.sandbox_id)
        )
        self.outcomes: List[ClaimOutcome] = response.outcomes


class TVCacheClient:
    """Typed async client for all TVCache server operations."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7878,
        timeout: float = 5,
        retries: int = 3,
        retry_backoff: float = 0.05,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        # One request/response at a time per connection; responses carry no
        # request id, so interleaving would hand a response to the wrong caller.
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self) -> "TVCacheClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def connect(self) -> None:
        last_error: Optional[BaseException] = None

        for attempt in range(self.retries + 1):
            try:
                self.reader, self.writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port),
                    timeout=self.timeout,
                )
                return
            except (OSError, asyncio.TimeoutError) as error:
                last_error = error
                if attempt == self.retries:
                    break
                await asyncio.sleep(self.retry_backoff * (attempt + 1))

        assert last_error is not None
        raise last_error

    async def close(self) -> None:
        # Detach synchronously. Cancellation during wait_closed must not leave
        # a partially consumed response available to the next request.
        writer = self.writer
        self.reader = None
        self.writer = None
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, asyncio.TimeoutError):
                pass

    async def _send_request(self, request: bytes) -> object:
        """Sends one request and decodes one response.

        A broken connection is discarded so the next request dials fresh; the
        failed request itself is not retried (a retried find would double-count
        a sandbox reference).

        Cancellation after a write also discards the connection: responses have
        no request IDs, so a late or partial response cannot safely be reused.
        The cancelled request is never replayed. Disconnecting an owner also
        fails its outstanding claims under the server's existing policy; this
        is not an exactly-once side-effect or unknown-lease recovery guarantee.
        """
        async with self._lock:
            if self.reader is None or self.writer is None:
                await self.connect()

            try:
                self.writer.write(request)
                await asyncio.wait_for(self.writer.drain(), timeout=self.timeout)
                return await asyncio.wait_for(
                    protocol.decode_response(self.reader),
                    timeout=self.timeout,
                )
            except (OSError, EOFError, asyncio.IncompleteReadError,
                    asyncio.TimeoutError, asyncio.CancelledError):
                await self.close()
                raise

    async def insert(
        self,
        task_id: str,
        path: List[str],
        value: Optional[str],
        sandbox_id: Optional[str],
        runtime: int,
        ttl: Optional[int] = None,
    ) -> InsertResult:
        response = await self._send_request(
            protocol.encode_insert(task_id, path, value, sandbox_id, runtime, ttl)
        )
        return self._expect(response, InsertResult)

    async def find(self, task_id: str, path: List[str]) -> FindResult:
        """Longest-prefix-match lookup returning a leased result.

        Use as `async with await client.find(task_id, path) as found:` so the
        reference taken by the server is always released.
        """
        return FindResult(self, task_id, await self.find_unmanaged(task_id, path))

    async def find_unmanaged(self, task_id: str, path: List[str]) -> FindResponse:
        """Raw find. If sandbox_id is set the server has already incremented its
        reference count and the caller owns the matching unref_sandbox call."""
        response = await self._send_request(protocol.encode_find(task_id, path))
        return self._expect(response, FindResponse)

    async def claim(self, task_id: str, prefix: List[str], entries: List[str]) -> ClaimResult:
        """Atomically resolves each entry under prefix as ready, pending (someone
        else is executing it), owner (this connection must execute and publish
        it), or missing (prefix absent). Entries must be distinct."""
        response = await self._send_request(protocol.encode_claim(task_id, prefix, entries))
        return ClaimResult(self, task_id, self._expect(response, ClaimResponse))

    async def inspect(
        self, task_id: str, prefix: List[str], entries: List[str], wait_ms: int = 0
    ) -> ClaimResponse:
        """Read-only status of each entry. With wait_ms > 0 the server holds the
        request until no entry is pending or wait_ms elapses (keep it below
        timeout). Never claims, creates nodes, or counts hits."""
        response = await self._send_request(
            protocol.encode_inspect(task_id, prefix, entries, wait_ms)
        )
        return self._expect(response, ClaimResponse)

    async def publish(
        self,
        task_id: str,
        path: List[str],
        generation: int,
        value: Optional[str],
        sandbox_id: Optional[str],
        runtime: int,
        ttl: Optional[int] = None,
    ) -> InsertResult:
        """Completes a claim this connection owns; value None marks it failed.
        Rejected (value_inserted False) if the claim is no longer this
        connection's current one."""
        response = await self._send_request(
            protocol.encode_publish(task_id, path, generation, value, sandbox_id, runtime, ttl)
        )
        return self._expect(response, InsertResult)

    async def unref_sandbox(self, task_id: str, sandbox_id: str) -> int:
        response = await self._send_request(
            protocol.encode_unref_sandbox(task_id, sandbox_id)
        )
        return self._expect(response, UnrefResult).remaining_references

    async def mark_sandboxes_evicted(self, records: List[SandboxEvictionRecord]) -> int:
        response = await self._send_request(
            protocol.encode_mark_sandboxes_evicted(records)
        )
        return self._expect(response, MarkResult).sandbox_ids_recorded

    async def list_sandbox_candidates(self, limit: int = 10) -> List[SandboxCandidate]:
        """Eviction candidates, ascending lpm-hit score (least useful first)."""
        response = await self._send_request(
            protocol.encode_list_sandbox_candidates(limit)
        )
        return self._expect(response, SandboxCandidates).candidates

    async def list_hot_sandboxes(self, limit: int = 10) -> List[SandboxCandidate]:
        """Fork-ahead candidates, descending lpm-hit score (most useful first)."""
        response = await self._send_request(protocol.encode_list_hot_sandboxes(limit))
        return self._expect(response, SandboxCandidates).candidates

    async def drain_reclaimable_sandboxes(self) -> List[ReclaimableSandbox]:
        """Takes ownership of every zero-reference victim sandbox. Destructive:
        only the single deletion loop (SandboxManager) should call this."""
        response = await self._send_request(
            protocol.encode_drain_reclaimable_sandboxes()
        )
        return self._expect(response, ReclaimableSandboxes).sandboxes

    async def report_fork_cost(self, task_id: str, fork_cost_ms: int) -> int:
        """Folds one measured env.fork() time into the task's server-side EMA
        and returns the updated estimate (shared across all workers)."""
        response = await self._send_request(
            protocol.encode_report_fork_cost(task_id, fork_cost_ms)
        )
        return self._expect(response, ForkCost).fork_cost_ms

    async def get_fork_cost(self, task_id: str) -> int:
        """The task's fork-cost EMA in milliseconds; 0 means no samples yet."""
        response = await self._send_request(protocol.encode_get_fork_cost(task_id))
        return self._expect(response, ForkCost).fork_cost_ms

    async def dump_all(self) -> dict:
        st = time.perf_counter()
        response = await self._send_request(protocol.encode_dump_all())
        et = time.perf_counter()
        self.logger.debug(f"dump_all took {et - st} seconds")
        return json.loads(self._expect(response, JsonResponse).json)

    def _expect(self, response: object, expected_type: type) -> object:
        if not isinstance(response, expected_type):
            raise TVCacheServerError(
                f"expected {expected_type.__name__}, got {type(response).__name__}"
            )
        return response
