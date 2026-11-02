"""Per-rollout tool call executor backed by the TVCache."""

import asyncio
import contextlib
import logging
import os
import time
from typing import List, Optional, Tuple, Type

from tvclient.client import FindResult, TVCacheClient
from tvclient.protocol import CLAIM_OWNER, CLAIM_PENDING, CLAIM_READY
from tvclient.sandbox import Sandbox
from tvclient.fork_bank import ROOT_ENV_ID, AbstractForkGenerator
from tvclient.snapshot_policy import (
    INITIAL_FORK_COST_SECONDS,
    SNAPSHOT_COST_MULTIPLIER,
    SnapshotPolicy,
)
from tvclient.tool_call import TestToolCall, ToolCall

MAX_RUNTIME_MS = 2**32
# Length of one Inspect long poll while waiting on another rollout's execution;
# must stay below the client's request timeout.
INSPECT_WAIT_MS = 2000


class TvcacheToolExecutor:
    """Executes tool calls for one rollout, reusing cached results and sandboxes.

    One executor per rollout, owning one cache connection and at most one live
    sandbox. The live sandbox is never discarded on a cache hit: it simply
    falls behind the rollout's history, and on a later miss it is either
    fast-forwarded or replaced by a fork of a deeper cached sandbox.

    With coalesce=True, concurrent rollouts that reach the same uncached call
    execute it once: the first claims it (a pending node) and the others wait
    for its result, which they then treat exactly like a cache hit.
    """

    def __init__(
        self,
        tool_call_env_class: Type[Sandbox],
        tool_call_class: Type[ToolCall],
        task_id: str,
        host: str = "127.0.0.1",
        port: int = 7878,
        adaptive: bool = True,
        fixed_threshold_seconds: float = 0.0,
        snapshot_cost_multiplier: float = SNAPSHOT_COST_MULTIPLIER,
        initial_fork_cost_seconds: float = INITIAL_FORK_COST_SECONDS,
        ttl: Optional[int] = None,
        coalesce: bool = False,
        max_wait_seconds: float = 1800.0,
    ):
        self.client = TVCacheClient(host, port)
        if coalesce and INSPECT_WAIT_MS / 1000 >= self.client.timeout:
            raise ValueError("INSPECT_WAIT_MS must be below the client request timeout")
        self.coalesce = coalesce
        # A waiter gives up and executes the call itself after this long, so a
        # hung owner can delay other rollouts but never strand them.
        self.max_wait_seconds = max_wait_seconds
        self.tool_call_env_class = tool_call_env_class
        self.tool_call_class = tool_call_class
        self.task_name = task_id
        self.ttl = ttl
        # Owns the un-checkpointed replay cost and the adaptive snapshot
        # threshold (backed by the server-side fork-cost EMA).
        self.snapshot_policy = SnapshotPolicy(
            self.client,
            task_id,
            adaptive,
            fixed_threshold_seconds,
            snapshot_cost_multiplier,
            initial_fork_cost_seconds,
        )
        self.rollout_id = str(time.time())
        self.rollout_sandbox: Optional[Sandbox] = None
        # Serialized stateful calls applied to the live sandbox; the invariant
        # is that this is a prefix of the current rollout's stateful chain.
        self.sandbox_history: List[str] = []
        self.total_calls = 0
        self.total_executions = 0
        self.total_waits = 0  # calls served by waiting on another rollout
        self.total_owned = 0  # claims this rollout executed for others too
        self.logger = logging.getLogger(__name__)
        self.fork_generator: Optional[AbstractForkGenerator] = None
        self._pending_stops: List[asyncio.Task] = []

    async def __aenter__(self) -> "TvcacheToolExecutor":
        await self.client.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Stops the live sandbox, waits for scheduled stops, closes the client."""
        await self._stop_rollout_environment()
        if self._pending_stops:
            await asyncio.gather(*self._pending_stops, return_exceptions=True)
            self._pending_stops = []
        await self.client.close()

    # ISSUE: if the caller doesn't call set_rollout_id, logging fails even to console, can we create a logger that just ignores the strings,
    def set_rollout_id(self, rollout_id: str) -> None:
        self.rollout_id = rollout_id
        self.logger = logging.getLogger(self.rollout_id)
        self.logger.setLevel(logging.DEBUG)

        os.makedirs("./rollouts", exist_ok=True)
        file_handler = logging.FileHandler(f"./rollouts/{rollout_id}.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def set_fork_bank(self, fork_bank: AbstractForkGenerator) -> None:
        self.fork_generator = fork_bank

    async def lookup_or_execute(self, tool_commands: List[ToolCall]) -> str:
        """Executes the last tool call unless the cache already has its value.

        The cache key is the stateful chain (calls whose will_mutate_state() is
        True) plus the last call; stateless intermediates never extend the key
        and are never replayed.
        """
        self.total_calls += 1
        history, current = tool_commands[:-1], tool_commands[-1]
        stateful_calls = [c for c in history if c.will_mutate_state()]
        prefix = [c.serialize() for c in stateful_calls]
        path = prefix + [current.serialize()]

        if self.coalesce:
            return await self._claim_or_execute(prefix, path, stateful_calls, current)
        return await self._find_or_execute(path, stateful_calls, current)

    async def _find_or_execute(
        self, path: List[str], stateful_calls: List[ToolCall], current: ToolCall
    ) -> str:
        cst = time.perf_counter()
        found = await self.client.find(self.task_name, path)
        cet = time.perf_counter()

        async with found:
            if found.value is not None:
                self.logger.debug(
                    f"CACHE HIT for task id {self.task_name} at depth {len(path)} "
                    f"in {cet - cst} seconds for rollout {self.rollout_id}"
                )
                return found.value

            self.logger.debug(
                f"CACHE MISS for task id {self.task_name} at depth {len(path)} "
                f"for rollout {self.rollout_id}"
            )
            # The fork (if any) happens while the lease is held, so the cached
            # sandbox cannot be reclaimed mid-fork.
            environment, resume_depth = await self._acquire_environment(found, path)

        suffix_calls = stateful_calls[resume_depth:] + [current]
        return await self._execute_and_insert(environment, path, suffix_calls, resume_depth)

    async def _claim_or_execute(
        self,
        prefix: List[str],
        path: List[str],
        stateful_calls: List[ToolCall],
        current: ToolCall,
    ) -> str:
        """Coalescing lookup: a ready result is a hit, a call another rollout is
        executing is awaited, and otherwise this rollout executes it (as the
        claim's owner, or uncoalesced when the prefix is not cached)."""
        async with await self.client.claim(self.task_name, prefix, [path[-1]]) as claimed:
            outcome = claimed.outcomes[0]

            if outcome.status == CLAIM_READY:
                self.logger.debug(
                    f"CACHE HIT for task id {self.task_name} at depth {len(path)} "
                    f"for rollout {self.rollout_id}"
                )
                return outcome.value

            if outcome.status != CLAIM_PENDING:
                generation = outcome.generation if outcome.status == CLAIM_OWNER else None
                if generation is not None:
                    self.total_owned += 1
                self.logger.debug(
                    f"CACHE {'OWNER' if generation is not None else 'MISS'} for task id "
                    f"{self.task_name} at depth {len(path)} for rollout {self.rollout_id}"
                )
                try:
                    # Under the lease, as in _find_or_execute.
                    environment, resume_depth = await self._acquire_environment(claimed, path)
                except BaseException:
                    await self._fail_claim(path, generation)
                    raise

        if outcome.status == CLAIM_PENDING:
            return await self._wait_or_execute(prefix, path, stateful_calls, current)

        suffix_calls = stateful_calls[resume_depth:] + [current]
        try:
            return await self._execute_and_insert(
                environment, path, suffix_calls, resume_depth, generation
            )
        except BaseException:
            await self._fail_claim(path, generation)
            raise

    async def _wait_or_execute(
        self,
        prefix: List[str],
        path: List[str],
        stateful_calls: List[ToolCall],
        current: ToolCall,
    ) -> str:
        """Waits for the rollout executing this call. Its result is used like a
        hit: the live sandbox falls behind and is caught up on a later miss."""
        self.total_waits += 1
        self.logger.debug(
            f"CACHE WAIT for task id {self.task_name} at depth {len(path)} "
            f"for rollout {self.rollout_id}"
        )
        value = (await self._wait_for_owners(prefix, [path[-1]]))[0]
        if value is not None:
            return value

        # The owner failed, disconnected, or exceeded max_wait_seconds.
        self.logger.debug(
            f"[WAIT]: No result from owner at depth {len(path)}; executing without "
            f"coalescing for rollout {self.rollout_id}"
        )
        return await self._find_or_execute(path, stateful_calls, current)

    async def _wait_for_owners(self, prefix: List[str], entries: List[str]) -> List[Optional[str]]:
        """Each entry's published value, or None if its owner failed or the
        wait exceeded max_wait_seconds."""
        deadline = time.monotonic() + self.max_wait_seconds
        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            response = await self.client.inspect(
                self.task_name, prefix, entries, wait_ms=max(0, min(INSPECT_WAIT_MS, remaining_ms))
            )
            if remaining_ms <= 0 or all(o.status != CLAIM_PENDING for o in response.outcomes):
                return [o.value if o.status == CLAIM_READY else None for o in response.outcomes]

    async def _fail_claim(self, path: List[str], generation: Optional[int]) -> None:
        """Releases this rollout's claim so its waiters execute the call
        themselves. Safe after a publish: the server rejects it then. Best
        effort: if the connection is gone, the server already failed the claim."""
        if generation is None:
            return
        with contextlib.suppress(Exception):
            await self.client.publish(self.task_name, path, generation, None, None, runtime=0)

    async def test(self, tool_call_history: List[ToolCall]) -> str:
        """Returns the (possibly cached) test result for the rollout's history.

        Terminal call: the rollout environment is stopped afterwards.
        """
        test_tool_call = TestToolCall("")
        result = await self.lookup_or_execute(tool_call_history + [test_tool_call])
        await self._stop_rollout_environment()
        return result

    async def _acquire_environment(
        self, found: FindResult, tool_call_list: List[str]
    ) -> Tuple[Sandbox, int]:
        """Picks the deepest valid resume point: forked LPM sandbox, the live
        sandbox (fast-forwarded), or a fresh environment."""
        live_usable = (
            self.rollout_sandbox is not None
            and tool_call_list[: len(self.sandbox_history)] == self.sandbox_history
        )
        live_depth = len(self.sandbox_history) if live_usable else 0

        # A sandbox cached at the full path already includes the current call's
        # mutation, so it can never be used to (re-)execute the current call.
        lpm_depth = 0
        if found.sandbox_id is not None:
            lpm_depth = len(found.sandbox_path)

        if found.sandbox_id is not None and lpm_depth > live_depth:
            lpm_sandbox = await self._fork_environment(found.sandbox_id)
            self._replace_rollout_environment(lpm_sandbox, list(tool_call_list[:lpm_depth]))
            self.logger.debug(
                f"[ENV]: Resuming from forked sandbox at depth {lpm_depth} "
                f"(live sandbox depth was {live_depth}) for rollout {self.rollout_id}"
            )
            return lpm_sandbox, lpm_depth

        if live_usable:
            if live_depth < len(tool_call_list) - 1:
                self.logger.debug(
                    f"[ENV]: Fast-forwarding {self.rollout_sandbox.get_id()} from depth "
                    f"{live_depth} to {len(tool_call_list) - 1} for rollout {self.rollout_id}"
                )
            return self.rollout_sandbox, live_depth

        environment = await self._fresh_environment()
        self._replace_rollout_environment(environment, [])
        return environment, 0

    async def _fork_environment(self, parent_env_id: str) -> Sandbox:
        """Forks a cached snapshot into a live sandbox to resume from it.

        Prefers a pre created copy from the fork bank; otherwise
        attaches to the stored snapshot handle and start()s it, which restores
        the snapshot into a fresh live sandbox.
        """
        if self.fork_generator is not None:
            bank_env_id = self.fork_generator.get_forked_env(self.task_name, parent_env_id)
            if bank_env_id is not None:
                self.logger.debug(
                    f"[ENV]: Using fork bank copy {bank_env_id} of parent {parent_env_id} "
                    f"for rollout {self.rollout_id}"
                )
                return self.tool_call_env_class(task_name=self.task_name, env_id=bank_env_id)

        lpm_sandbox = self.tool_call_env_class(
            task_name=self.task_name, env_id=parent_env_id
        )
        st = time.perf_counter()
        await lpm_sandbox.start()
        et = time.perf_counter()
        await self.snapshot_policy.observe_fork(et - st)
        self.logger.debug(
            f"[ENV]: Forked {lpm_sandbox.get_id()} from parent {parent_env_id} "
            f"in {et - st:.2f} seconds for rollout {self.rollout_id}"
        )
        return lpm_sandbox

    async def _fresh_environment(self) -> Sandbox:
        if self.fork_generator is not None:
            bank_env_id = self.fork_generator.get_forked_env(self.task_name, ROOT_ENV_ID)
            if bank_env_id is not None:
                self.logger.debug(
                    f"[ENV]: Using warm root environment {bank_env_id} for rollout {self.rollout_id}"
                )
                return self.tool_call_env_class(task_name=self.task_name, env_id=bank_env_id)

        environment = self.tool_call_env_class(task_name=self.task_name)
        st = time.perf_counter()
        await environment.start()
        et = time.perf_counter()
        self.logger.debug(
            f"[ENV]: Started fresh environment {environment.get_id()} "
            f"in {et - st:.2f} seconds for rollout {self.rollout_id}"
        )
        return environment

    async def _execute_and_insert(
        self,
        environment: Sandbox,
        path: List[str],
        suffix_calls: List[ToolCall],
        resume_depth: int,
        generation: Optional[int] = None,
    ) -> str:
        """Executes the unmatched suffix in the environment and populates the cache.

        With a generation, this rollout owns the claim on the last call (the
        current one) and publishes it; the replayed calls before it are already
        cached and are inserted as usual."""
        last_value = None
        fork_cost_ms = await self.snapshot_policy.prime()
        if fork_cost_ms is not None:
            self.logger.debug(
                f"[ENV]: Server fork cost for task {self.task_name} is "
                f"{fork_cost_ms} ms for rollout {self.rollout_id}"
            )

        for offset, tool_call in enumerate(suffix_calls):
            depth = resume_depth + offset + 1
            self.total_executions += 1

            st = time.perf_counter()
            try:
                value = await environment.execute(tool_call)
            except Exception:
                # The sandbox state is unknown after a raised execution.
                self.logger.debug(
                    f"[TOOL EXEC]: Execution raised; discarding environment "
                    f"{environment.get_id()} for rollout {self.rollout_id}"
                )
                self._schedule_stop(environment)
                self.rollout_sandbox = None
                self.sandbox_history = []
                self.snapshot_policy.reset()
                raise
            et = time.perf_counter()
            runtime_seconds = et - st
            # Replay would re-pay this call (failed ones included), so it
            # always counts toward the un-checkpointed replay cost.
            self.snapshot_policy.record_execution(runtime_seconds)
            self.logger.debug(
                f"[TOOL EXEC]: Executed tool call {tool_call.serialize()} "
                f"in {runtime_seconds} seconds for rollout {self.rollout_id}"
            )

            if tool_call.will_mutate_state():
                self.sandbox_history.append(tool_call.serialize())

            claimed = generation if depth == len(path) else None

            if value.startswith("Failed to execute"):
                # Environment-reported failure: keep the rollout going but never cache it.
                self.logger.debug(
                    f"[TOOL EXEC]: Not caching failed result at depth {depth} "
                    f"for rollout {self.rollout_id}"
                )
                await self._fail_claim(path, claimed)
                last_value = value
                continue
            
            await self._update_tvcache_with_execution(
                tool_call, environment, path, depth, value, runtime_seconds, claimed
            )
            last_value = value

        return last_value

    async def _update_tvcache_with_execution(
        self,
        tool_call: ToolCall,
        environment: Sandbox,
        path: List[str],
        depth: int,
        value: str,
        runtime_seconds: float,
        generation: Optional[int] = None,
    ) -> None:

        sandbox_id = None
        
        if tool_call.will_mutate_state() and self.snapshot_policy.should_snapshot():
            unsnapshotted_runtime = self.snapshot_policy.unsnapshotted_runtime
            snapshot_threshold = self.snapshot_policy.threshold()
            fst = time.perf_counter()
            stored_copy = await environment.fork()
            fet = time.perf_counter()
            await self.snapshot_policy.observe_fork(fet - fst)
            sandbox_id = stored_copy.get_id()
            self.logger.debug(
                f"[ENV]: Storing fork {sandbox_id} after {unsnapshotted_runtime:.2f} "
                f"seconds of un-checkpointed runtime (threshold {snapshot_threshold:.2f}, "
                f"fork took {fet - fst:.2f} seconds) for rollout {self.rollout_id}"
            )
            # Even if the insert below loses the sandbox race, the state is
            # checkpointed (by the winner), so the replay debt is paid.
            self.snapshot_policy.reset()

        ist = time.perf_counter()
        if generation is None:
            insert_result = await self.client.insert(
                self.task_name,
                list(path[:depth]),
                value,
                sandbox_id,
                runtime=self._runtime_ms(runtime_seconds),
                ttl=self.ttl,
            )
        else:
            # Completing our claim wakes every rollout waiting on this call.
            insert_result = await self.client.publish(
                self.task_name,
                list(path[:depth]),
                generation,
                value,
                sandbox_id,
                runtime=self._runtime_ms(runtime_seconds),
                ttl=self.ttl,
            )
            if not insert_result.value_inserted:
                # Our claim was failed meanwhile (e.g. the connection dropped);
                # the rollout still returns its own result.
                self.logger.debug(
                    f"Publish rejected at depth {depth} for rollout {self.rollout_id}"
                )
        iet = time.perf_counter()
        self.logger.debug(
            f"Inserted value at depth {depth} in {iet - ist} seconds "
            f"for rollout {self.rollout_id}"
        )

        if sandbox_id is not None and not insert_result.sandbox_inserted:
            # Another rollout won the race to attach a sandbox at this node;
            # our copy is unknown to the cache and would otherwise leak.
            self.logger.debug(
                f"[ENV]: Discarding orphan fork {sandbox_id}: node already had a sandbox"
            )
            self._schedule_stop(
                self.tool_call_env_class(task_name=self.task_name, env_id=sandbox_id)
            )

    def _replace_rollout_environment(
        self, environment: Sandbox, sandbox_history: List[str]
    ) -> None:
        old_environment = self.rollout_sandbox
        if old_environment is not None and old_environment is not environment:
            self.logger.debug(
                f"[ENV]: Replacing rollout environment {old_environment.get_id()} "
                f"for rollout {self.rollout_id}"
            )
            self._schedule_stop(old_environment)
        self.rollout_sandbox = environment
        self.sandbox_history = sandbox_history
        # A fork's resume point is a checkpoint; a fresh env's empty state
        # costs nothing — replay costs re-accumulate as they are paid.
        self.snapshot_policy.reset()

    def _schedule_stop(self, environment: Sandbox) -> None:
        """Stops an environment off the critical path."""

        async def _stop() -> None:
            try:
                await environment.stop()
            except Exception as error:
                self.logger.debug(
                    f"[ENV]: Failed to stop environment {environment.get_id()} due to {error}"
                )

        self._pending_stops = [task for task in self._pending_stops if not task.done()]
        self._pending_stops.append(asyncio.create_task(_stop()))

    async def _stop_rollout_environment(self) -> None:
        if self.rollout_sandbox is None:
            return
        environment = self.rollout_sandbox
        self.rollout_sandbox = None
        self.sandbox_history = []
        self.logger.debug(
            f"[ENV]: Stopping rollout environment {environment.get_id()} "
            f"for rollout {self.rollout_id}"
        )
        try:
            await environment.stop()
        except Exception as error:
            self.logger.debug(
                f"[ENV]: Failed to stop environment {environment.get_id()} due to {error}"
            )

    def _runtime_ms(self, runtime_seconds: float) -> int:
        return min(MAX_RUNTIME_MS, max(0, int(runtime_seconds * 1000)))
