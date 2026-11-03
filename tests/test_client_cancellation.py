"""Socket-free cancellation regressions for the actual TVCacheClient.

StreamReader and scripted writer fixtures provide bytes/events only, not a
cache implementation. No server, tools, sandbox, model, or GPU is exercised.
Run from the client checkout: python -m unittest discover -s tests -v
"""

import asyncio
import struct
import unittest
from unittest.mock import AsyncMock, patch

from tvclient import TVCacheClient, protocol


def claim_response(value="late cancelled result"):
    return (
        bytes([protocol.CLAIM_RESULTS_RESPONSE, 0, 0])
        + struct.pack(">H", 1)
        + bytes([protocol.CLAIM_READY])
        + struct.pack(">QI", 7, 1)
        + bytes([1])
        + protocol.write_string_u32(value)
    )


def fork_response(value):
    return bytes([protocol.FORK_COST_RESPONSE]) + struct.pack(">I", value)


class ObservedReader(asyncio.StreamReader):
    """Expose a real decoder's blocking read without sleeps or patching it."""

    def __init__(self):
        super().__init__()
        self.blocked = asyncio.Event()
        self.blocked_size = None

    async def readexactly(self, count):
        if len(self._buffer) < count:
            self.blocked_size = count
            self.blocked.set()
        return await super().readexactly(count)


class ScriptedWriter:
    def __init__(self, *, hold_drain=False, hold_close=False):
        self.requests = []
        self.drain_started = asyncio.Event()
        self.drain_release = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.closed = False
        if not hold_drain:
            self.drain_release.set()
        if not hold_close:
            self.close_release.set()

    def write(self, request):
        if self.closed:
            raise AssertionError("A closed writer must never be reused")
        self.requests.append(request)

    async def drain(self):
        self.drain_started.set()
        await self.drain_release.wait()

    def close(self):
        self.closed = True
        self.close_started.set()

    async def wait_closed(self):
        await self.close_release.wait()


class ClientCancellationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clients = []
        self.writers = []

    async def asyncTearDown(self):
        for writer in self.writers:
            writer.drain_release.set()
            writer.close_release.set()
        for client in self.clients:
            await asyncio.wait_for(client.close(), 1)

    def connection(self, *, hold_drain=False, hold_close=False):
        reader = ObservedReader()
        writer = ScriptedWriter(hold_drain=hold_drain, hold_close=hold_close)
        self.writers.append(writer)
        return reader, writer

    def client(self, reader, writer):
        client = TVCacheClient(timeout=1, retries=3)
        client.reader, client.writer = reader, writer
        self.clients.append(client)
        return client

    async def inspect_then_cancel(self, client, reader):
        request = asyncio.create_task(client.inspect("fixture-task", [], ["held"], wait_ms=500))
        await asyncio.wait_for(reader.blocked.wait(), 1)
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request

    async def assert_fresh_followup(self, client, old_reader, old_writer):
        fresh_reader, fresh_writer = self.connection()
        fresh_reader.feed_data(fork_response(4321))
        dial = AsyncMock(return_value=(fresh_reader, fresh_writer))
        with patch("tvclient.client.asyncio.open_connection", dial):
            # Without cleanup this reads the cancelled Inspect's late response
            # and fails with "expected ForkCost, got ClaimResponse" (or loses
            # framing when cancellation happened partway through decoding).
            self.assertEqual(await client.get_fork_cost("followup-task"), 4321)
        dial.assert_awaited_once_with(client.host, client.port)
        self.assertTrue(old_writer.closed)
        self.assertIs(client.reader, fresh_reader)
        self.assertIs(client.writer, fresh_writer)
        self.assertEqual(len(old_writer.requests), 1, "Cancelled request must not be replayed")
        self.assertEqual(fresh_writer.requests, [protocol.encode_get_fork_cost("followup-task")])

    async def test_cancelled_inspect_discards_late_response_and_reconnects(self):
        reader, writer = self.connection()
        client = self.client(reader, writer)
        await self.inspect_then_cancel(client, reader)
        reader.feed_data(claim_response())
        await self.assert_fresh_followup(client, reader, writer)

    async def test_cancel_during_partial_response_does_not_reuse_broken_framing(self):
        reader, writer = self.connection()
        client = self.client(reader, writer)
        response = claim_response()
        # The real decoder consumes tag/flags/count/status, then waits for the
        # remaining bytes of the u64 node ID. Its old framing is unrecoverable.
        reader.feed_data(response[:8])
        await self.inspect_then_cancel(client, reader)
        self.assertEqual(reader.blocked_size, 8)
        reader.feed_data(response[8:])
        await self.assert_fresh_followup(client, reader, writer)

    async def test_cancel_during_drain_closes_without_retrying_the_request(self):
        reader, writer = self.connection(hold_drain=True)
        client = self.client(reader, writer)
        request = asyncio.create_task(client.inspect("fixture-task", [], ["held"], wait_ms=500))
        await asyncio.wait_for(writer.drain_started.wait(), 1)
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request
        self.assertTrue(writer.closed)
        self.assertIsNone(client.reader)
        self.assertIsNone(client.writer)
        self.assertEqual(writer.requests, [protocol.encode_inspect("fixture-task", [], ["held"], 500)])
        self.assertFalse(reader.blocked.is_set(), "Decoder should not have started before drain completed")

    async def test_queued_cancellation_does_not_close_the_active_requests_connection(self):
        reader, writer = self.connection()
        client = self.client(reader, writer)
        active = asyncio.create_task(client.get_fork_cost("active"))
        await asyncio.wait_for(reader.blocked.wait(), 1)
        queued_started = asyncio.Event()

        async def queued_request():
            queued_started.set()
            return await client.get_fork_cost("queued")

        queued = asyncio.create_task(queued_request())
        await asyncio.wait_for(queued_started.wait(), 1)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await queued
        self.assertFalse(writer.closed)
        self.assertIs(client.writer, writer)
        self.assertEqual(writer.requests, [protocol.encode_get_fork_cost("active")])
        reader.feed_data(fork_response(23))
        self.assertEqual(await active, 23)
        reader.feed_data(fork_response(24))
        self.assertEqual(await client.get_fork_cost("later"), 24)
        self.assertFalse(writer.closed)
        self.assertEqual(writer.requests, [protocol.encode_get_fork_cost("active"),
                                           protocol.encode_get_fork_cost("later")])

    async def test_repeat_cancellation_during_close_cannot_retain_stale_stream(self):
        reader, writer = self.connection(hold_close=True)
        client = self.client(reader, writer)
        request = asyncio.create_task(client.inspect("fixture-task", [], ["held"], wait_ms=500))
        await asyncio.wait_for(reader.blocked.wait(), 1)
        request.cancel()
        try:
            await asyncio.wait_for(writer.close_started.wait(), 1)
            self.assertIsNone(client.reader, "Detach before waiting for transport shutdown")
            self.assertIsNone(client.writer, "Repeat cancellation must never expose old writer")
        finally:
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
        reader.feed_data(claim_response())
        await self.assert_fresh_followup(client, reader, writer)

    async def test_close_detaches_before_awaiting_shutdown(self):
        reader, writer = self.connection(hold_close=True)
        client = self.client(reader, writer)
        closing = asyncio.create_task(client.close())
        await asyncio.wait_for(writer.close_started.wait(), 1)
        try:
            self.assertTrue(writer.closed)
            self.assertIsNone(client.reader)
            self.assertIsNone(client.writer)
        finally:
            writer.close_release.set()
            await closing

    async def test_completed_response_keeps_healthy_connection(self):
        reader, writer = self.connection()
        client = self.client(reader, writer)
        reader.feed_data(fork_response(19) + fork_response(20))
        self.assertEqual(await client.get_fork_cost("one"), 19)
        self.assertEqual(await client.get_fork_cost("two"), 20)
        self.assertIs(client.reader, reader)
        self.assertIs(client.writer, writer)
        self.assertFalse(writer.closed)


if __name__ == "__main__":
    unittest.main()
