"""Binary wire protocol for the TVCache server.

Request op bytes 1-13 and response type bytes 1-10, big-endian, with
bool-prefixed optional fields. This module only knows how to encode request
payloads and decode responses from an asyncio.StreamReader; connection
handling lives in TVCacheClient.
"""

import asyncio
import struct
from dataclasses import dataclass
from typing import List, Optional


INSERT_REQUEST = 1
FIND_REQUEST = 2
DUMP_ALL_REQUEST = 3
UNREF_SANDBOX_REQUEST = 4
MARK_SANDBOXES_EVICTED_REQUEST = 5
LIST_SANDBOX_CANDIDATES_REQUEST = 6
DRAIN_RECLAIMABLE_SANDBOXES_REQUEST = 7
LIST_HOT_SANDBOXES_REQUEST = 8
REPORT_FORK_COST_REQUEST = 9
GET_FORK_COST_REQUEST = 10
CLAIM_REQUEST = 11
INSPECT_REQUEST = 12
PUBLISH_REQUEST = 13

INSERT_RESULT_RESPONSE = 1
FIND_RESULT_RESPONSE = 2
ERROR_RESPONSE = 3
JSON_RESPONSE = 4
UNREF_SANDBOX_RESULT_RESPONSE = 5
MARK_SANDBOXES_EVICTED_RESULT_RESPONSE = 6
SANDBOX_CANDIDATES_RESPONSE = 7
RECLAIMABLE_SANDBOXES_RESPONSE = 8
FORK_COST_RESPONSE = 9
CLAIM_RESULTS_RESPONSE = 10

# Per-entry status of a claim or inspect.
CLAIM_MISSING = 0  # prefix absent, or the node has no result yet
CLAIM_OWNER = 1  # this connection must execute the call and publish it
CLAIM_PENDING = 2  # another connection is executing it
CLAIM_READY = 3  # cached result available
CLAIM_FAILED = 4  # the owner's execution failed or its connection closed


class TVCacheServerError(Exception):
    """Raised when the server answers a request with an error response."""


@dataclass(frozen=True)
class InsertResult:
    value_inserted: bool
    sandbox_inserted: bool


@dataclass(frozen=True)
class FindResponse:
    value: Optional[str]
    sandbox_path: Optional[List[str]]
    sandbox_id: Optional[str]


@dataclass(frozen=True)
class UnrefResult:
    remaining_references: int


@dataclass(frozen=True)
class MarkResult:
    sandbox_ids_recorded: int


@dataclass(frozen=True)
class SandboxCandidate:
    task_id: str
    sandbox_path: List[str]
    sandbox_id: str
    score: int


@dataclass(frozen=True)
class SandboxCandidates:
    candidates: List[SandboxCandidate]


@dataclass(frozen=True)
class ReclaimableSandbox:
    task_id: str
    sandbox_id: str


@dataclass(frozen=True)
class ReclaimableSandboxes:
    sandboxes: List[ReclaimableSandbox]


@dataclass(frozen=True)
class SandboxEvictionRecord:
    task_id: str
    sandbox_path: List[str]
    sandbox_id: str


@dataclass(frozen=True)
class ForkCost:
    fork_cost_ms: int


@dataclass(frozen=True)
class JsonResponse:
    json: str


@dataclass(frozen=True)
class ClaimOutcome:
    status: int
    node_id: int
    generation: int
    value: Optional[str]  # only for CLAIM_READY


@dataclass(frozen=True)
class ClaimResponse:
    """One outcome per requested entry, plus the prefix sandbox lease (claim only)."""

    sandbox_path: Optional[List[str]]
    sandbox_id: Optional[str]
    outcomes: List[ClaimOutcome]


def write_u8(value: int) -> bytes:
    return struct.pack(">B", value)


def write_bool(value: bool) -> bytes:
    return write_u8(1 if value else 0)


def write_u16(value: int) -> bytes:
    return struct.pack(">H", value)


def write_u32(value: int) -> bytes:
    return struct.pack(">I", value)


def write_string_u16(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return write_u16(len(encoded)) + encoded


def write_string_u32(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return write_u32(len(encoded)) + encoded


def write_path(path: List[str]) -> bytes:
    encoded = write_u16(len(path))
    for entry in path:
        encoded += write_string_u16(entry)
    return encoded


def encode_insert(
    task_id: str,
    path: List[str],
    value: Optional[str],
    sandbox_id: Optional[str],
    runtime: int,
    ttl: Optional[int] = None,
) -> bytes:
    encoded = write_u8(INSERT_REQUEST)
    encoded += write_string_u16(task_id)
    encoded += write_path(path)
    return encoded + _encode_insert_fields(value, sandbox_id, runtime, ttl)


def _encode_insert_fields(
    value: Optional[str], sandbox_id: Optional[str], runtime: int, ttl: Optional[int]
) -> bytes:
    """The fields Insert and Publish share, after the path (and generation)."""
    encoded = write_u32(runtime)  # milliseconds

    encoded += write_bool(value is not None)
    if value is not None:
        encoded += write_string_u32(value)

    encoded += write_bool(sandbox_id is not None)
    if sandbox_id is not None:
        encoded += write_string_u16(sandbox_id)

    encoded += write_bool(ttl is not None)
    if ttl is not None:
        encoded += write_u32(ttl)

    return encoded


def encode_publish(
    task_id: str,
    path: List[str],
    generation: int,
    value: Optional[str],
    sandbox_id: Optional[str],
    runtime: int,
    ttl: Optional[int] = None,
) -> bytes:
    """An insert carrying the claim's generation; value None reports failure."""
    return (
        write_u8(PUBLISH_REQUEST)
        + write_string_u16(task_id)
        + write_path(path)
        + write_u32(generation)
        + _encode_insert_fields(value, sandbox_id, runtime, ttl)
    )


def encode_claim(task_id: str, prefix: List[str], entries: List[str]) -> bytes:
    return (
        write_u8(CLAIM_REQUEST)
        + write_string_u16(task_id)
        + write_path(prefix)
        + write_path(entries)
    )


def encode_inspect(task_id: str, prefix: List[str], entries: List[str], wait_ms: int) -> bytes:
    return (
        write_u8(INSPECT_REQUEST)
        + write_string_u16(task_id)
        + write_path(prefix)
        + write_path(entries)
        + write_u32(wait_ms)
    )


def encode_find(task_id: str, path: List[str]) -> bytes:
    return write_u8(FIND_REQUEST) + write_string_u16(task_id) + write_path(path)


def encode_dump_all() -> bytes:
    return write_u8(DUMP_ALL_REQUEST)


def encode_unref_sandbox(task_id: str, sandbox_id: str) -> bytes:
    return (
        write_u8(UNREF_SANDBOX_REQUEST)
        + write_string_u16(task_id)
        + write_string_u16(sandbox_id)
    )


def encode_mark_sandboxes_evicted(records: List[SandboxEvictionRecord]) -> bytes:
    encoded = write_u8(MARK_SANDBOXES_EVICTED_REQUEST) + write_u16(len(records))
    for record in records:
        encoded += write_string_u16(record.task_id)
        encoded += write_path(record.sandbox_path)
        encoded += write_string_u16(record.sandbox_id)
    return encoded


def encode_list_sandbox_candidates(limit: int = 10) -> bytes:
    return write_u8(LIST_SANDBOX_CANDIDATES_REQUEST) + write_u16(limit)


def encode_drain_reclaimable_sandboxes() -> bytes:
    return write_u8(DRAIN_RECLAIMABLE_SANDBOXES_REQUEST)


def encode_list_hot_sandboxes(limit: int = 10) -> bytes:
    return write_u8(LIST_HOT_SANDBOXES_REQUEST) + write_u16(limit)


def encode_report_fork_cost(task_id: str, fork_cost_ms: int) -> bytes:
    return (
        write_u8(REPORT_FORK_COST_REQUEST)
        + write_string_u16(task_id)
        + write_u32(fork_cost_ms)
    )


def encode_get_fork_cost(task_id: str) -> bytes:
    return write_u8(GET_FORK_COST_REQUEST) + write_string_u16(task_id)


async def read_u8(reader: asyncio.StreamReader) -> int:
    return struct.unpack(">B", await reader.readexactly(1))[0]


async def read_bool(reader: asyncio.StreamReader) -> bool:
    value = await read_u8(reader)
    if value == 0:
        return False
    if value == 1:
        return True
    raise ValueError(f"invalid bool byte: {value}")


async def read_u16(reader: asyncio.StreamReader) -> int:
    return struct.unpack(">H", await reader.readexactly(2))[0]


async def read_u32(reader: asyncio.StreamReader) -> int:
    return struct.unpack(">I", await reader.readexactly(4))[0]


async def read_u64(reader: asyncio.StreamReader) -> int:
    return struct.unpack(">Q", await reader.readexactly(8))[0]


async def read_string_u16(reader: asyncio.StreamReader) -> str:
    return (await reader.readexactly(await read_u16(reader))).decode("utf-8")


async def read_string_u32(reader: asyncio.StreamReader) -> str:
    return (await reader.readexactly(await read_u32(reader))).decode("utf-8")


async def read_path(reader: asyncio.StreamReader) -> List[str]:
    count = await read_u16(reader)
    return [await read_string_u16(reader) for _ in range(count)]


async def read_optional_string_u16(reader: asyncio.StreamReader) -> Optional[str]:
    if await read_bool(reader):
        return await read_string_u16(reader)
    return None


async def read_optional_string_u32(reader: asyncio.StreamReader) -> Optional[str]:
    if await read_bool(reader):
        return await read_string_u32(reader)
    return None


async def read_optional_path(reader: asyncio.StreamReader) -> Optional[List[str]]:
    if await read_bool(reader):
        return await read_path(reader)
    return None


async def read_sandbox_candidate(reader: asyncio.StreamReader) -> SandboxCandidate:
    return SandboxCandidate(
        task_id=await read_string_u16(reader),
        sandbox_path=await read_path(reader),
        sandbox_id=await read_string_u16(reader),
        score=await read_u64(reader),
    )


async def decode_response(reader: asyncio.StreamReader) -> object:
    """Decodes one response into its dataclass; raises TVCacheServerError on type 3."""
    response_type = await read_u8(reader)

    if response_type == INSERT_RESULT_RESPONSE:
        return InsertResult(
            value_inserted=await read_bool(reader),
            sandbox_inserted=await read_bool(reader),
        )

    if response_type == FIND_RESULT_RESPONSE:
        return FindResponse(
            value=await read_optional_string_u32(reader),
            sandbox_path=await read_optional_path(reader),
            sandbox_id=await read_optional_string_u16(reader),
        )

    if response_type == ERROR_RESPONSE:
        raise TVCacheServerError(await read_string_u32(reader))

    if response_type == JSON_RESPONSE:
        return JsonResponse(json=await read_string_u32(reader))

    if response_type == UNREF_SANDBOX_RESULT_RESPONSE:
        return UnrefResult(remaining_references=await read_u32(reader))

    if response_type == MARK_SANDBOXES_EVICTED_RESULT_RESPONSE:
        return MarkResult(sandbox_ids_recorded=await read_u32(reader))

    if response_type == SANDBOX_CANDIDATES_RESPONSE:
        count = await read_u16(reader)
        return SandboxCandidates(
            candidates=[await read_sandbox_candidate(reader) for _ in range(count)]
        )

    if response_type == RECLAIMABLE_SANDBOXES_RESPONSE:
        count = await read_u32(reader)
        return ReclaimableSandboxes(
            sandboxes=[
                ReclaimableSandbox(
                    task_id=await read_string_u16(reader),
                    sandbox_id=await read_string_u16(reader),
                )
                for _ in range(count)
            ]
        )

    if response_type == FORK_COST_RESPONSE:
        return ForkCost(fork_cost_ms=await read_u32(reader))

    if response_type == CLAIM_RESULTS_RESPONSE:
        sandbox_path = await read_optional_path(reader)
        sandbox_id = await read_optional_string_u16(reader)
        count = await read_u16(reader)
        return ClaimResponse(
            sandbox_path=sandbox_path,
            sandbox_id=sandbox_id,
            outcomes=[
                ClaimOutcome(
                    status=await read_u8(reader),
                    node_id=await read_u64(reader),
                    generation=await read_u32(reader),
                    value=await read_optional_string_u32(reader),
                )
                for _ in range(count)
            ],
        )

    raise ValueError(f"unknown response type: {response_type}")
