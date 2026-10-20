"""Observable lifecycle invariants; no model downloads or GPU are needed."""

import asyncio
import unittest

from audri.audio import AudioFormat, AudioFrame
from audri.speak_tool import SpeakTool
from audri.tts.audio_egress import CallbackAudioEgress, NullAudioEgress, QueueAudioEgress
from audri.tts.chunker import ChunkerConfig
from audri.tts.driver import TTSDriver
from audri.tts.session import (
    SessionClosedError, SessionConfig, SessionLimitError, SessionState, SpeechSession,
)
from audri.tts.types import SynthesisRequest


FORMAT = AudioFormat(44_100)
FRAME = AudioFrame(b"\x01\x00" * 882, FORMAT)
CHUNKER = ChunkerConfig(min_chars_to_start=1, max_chars_per_chunk=32, lookahead_chars=31)


class RecordingDriver(TTSDriver):
    def __init__(self, frames=1, delay=0):
        self.requests = []
        self.tokens = []
        self.frames = frames
        self.delay = delay
        self.closed_streams = 0
        self.closed = False
        self.started = asyncio.Event()

    async def stream(self, request, *, cancel=None):
        self.requests.append(request)
        self.tokens.append(cancel)
        self.started.set()
        try:
            for _ in range(self.frames):
                if self.delay:
                    await asyncio.sleep(self.delay)
                if cancel is not None:
                    cancel.raise_if_cancelled()
                yield FRAME
        finally:
            self.closed_streams += 1

    async def aclose(self):
        self.closed = True


class FailingDriver(RecordingDriver):
    async def stream(self, request, *, cancel=None):
        await asyncio.sleep(0)
        raise OSError("upstream unavailable")
        yield FRAME


class GatedDriver(RecordingDriver):
    def __init__(self):
        super().__init__()
        self.release = asyncio.Event()

    async def stream(self, request, *, cancel=None):
        self.started.set()
        try:
            await self.release.wait()
            yield FRAME
        finally:
            self.closed_streams += 1


def session(driver=None, sink=None, **kwargs):
    return SpeechSession(
        "test", "default", "English", driver or RecordingDriver(),
        sink or NullAudioEgress(), chunker_config=CHUNKER, **kwargs,
    )


async def wait_for_blocked_put(item):
    async with asyncio.timeout(1):
        while not (
            item._delta_queue.full()
            and any(not task.done() for task in item._pending_puts)
        ):
            await asyncio.sleep(.001)


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordered_bounded_chunks_and_metrics(self):
        driver = RecordingDriver()
        sink = NullAudioEgress()
        item = session(driver, sink)
        item.start()
        text = "First sentence. Second sentence. " + "word " * 60
        await item.push_delta(text)
        metrics = await asyncio.wait_for(item.finish(), 2)
        self.assertEqual(" ".join(r.text for r in driver.requests).split(), text.split())
        self.assertTrue(all(len(r.text) <= 32 for r in driver.requests))
        self.assertEqual(metrics.state, "done")
        self.assertEqual(metrics.input_chars, len(text))
        self.assertEqual(metrics.frames, len(driver.requests))
        self.assertAlmostEqual(metrics.audio_seconds, metrics.frames * .02)
        self.assertEqual(sink.format.sample_rate, 44_100)
        self.assertTrue(sink.closed)
        self.assertIsNotNone(metrics.first_audio_ms)
        self.assertGreaterEqual(metrics.rtf, 0)
        self.assertIn("rtf", metrics.as_dict())

    async def test_empty_and_repeated_finish_are_safe(self):
        driver = RecordingDriver()
        item = session(driver)
        item.start()
        first = await asyncio.wait_for(item.finish(), 1)
        second = await item.finish()
        await item.push_delta("", is_final=True)
        self.assertIs(first, second)
        self.assertEqual(driver.requests, [])
        self.assertEqual(first.frames, 0)
        self.assertIsNone(first.rtf)

    async def test_delta_after_final_is_rejected(self):
        item = session(RecordingDriver(delay=.02))
        item.start()
        await item.push_delta("hello", is_final=True)
        with self.assertRaises(SessionClosedError):
            await item.push_delta("late data")
        await item.wait()

    async def test_template_controls_are_forwarded(self):
        driver = RecordingDriver()
        request = SynthesisRequest("ignored", reference_id="narrator", seed=42, temperature=.7)
        item = session(driver, request=request)
        item.start()
        await item.push_delta("First sentence. Second sentence.")
        await item.finish()
        self.assertTrue(driver.requests)
        for actual in driver.requests:
            self.assertEqual(actual.seed, 42)
            self.assertEqual(actual.reference_id, "narrator")
            self.assertEqual(actual.temperature, .7)
            self.assertNotEqual(actual.text, "ignored")

    async def test_timeout_commits_without_end(self):
        driver = RecordingDriver()
        item = SpeechSession(
            "timeout", "default", "English", driver, NullAudioEgress(),
            chunker_config=ChunkerConfig(boundary_timeout_ms=15),
        )
        item.start()
        await item.push_delta("hello there")
        await asyncio.wait_for(driver.started.wait(), 1)
        self.assertEqual(driver.requests[0].text, "hello there")
        await item.finish()

    async def test_driver_error_reaches_caller_and_aborts_output(self):
        sink = NullAudioEgress()
        item = session(FailingDriver(), sink)
        item.start()
        await item.push_delta("A sentence.")
        with self.assertRaisesRegex(OSError, "upstream unavailable"):
            await asyncio.wait_for(item.finish(), 1)
        self.assertEqual(item.state, SessionState.FAILED)
        self.assertTrue(sink.aborted)
        self.assertIn("OSError", item.metrics.error)

    async def test_sink_failure_cancels_generator(self):
        driver = RecordingDriver(frames=100, delay=.001)

        async def fail(frame):
            raise OSError("speaker disconnected")

        sink = CallbackAudioEgress(fail)
        item = session(driver, sink)
        item.start()
        await item.push_delta("Audio.")
        with self.assertRaisesRegex(OSError, "speaker disconnected"):
            await asyncio.wait_for(item.finish(), 1)
        self.assertEqual(driver.closed_streams, 1)
        self.assertTrue(sink.aborted)

    async def test_immediate_cancel_before_runner_starts(self):
        sink = NullAudioEgress()
        item = session(sink=sink)
        item.start()
        metrics = await asyncio.wait_for(item.cancel(), 1)
        self.assertEqual(metrics.state, "cancelled")
        self.assertTrue(sink.aborted)
        self.assertIs(await item.wait(), metrics)
        self.assertIs(await item.cancel(), metrics)

    async def test_cancel_clears_buffered_audio_and_closes_generator(self):
        driver = RecordingDriver(frames=10_000)
        sink = QueueAudioEgress(maxsize=1)
        item = session(driver, sink, config=SessionConfig(audio_queue_size=2))
        item.start()
        await item.push_delta("Audio.")
        await asyncio.wait_for(driver.started.wait(), 1)
        await asyncio.sleep(.02)
        metrics = await asyncio.wait_for(item.cancel(), 1)
        self.assertEqual(metrics.state, "cancelled")
        self.assertEqual(sink.queue.qsize(), 0)
        self.assertTrue(sink.aborted)
        self.assertEqual(driver.closed_streams, 1)
        self.assertLessEqual(metrics.audio_queue_highwater, 2)
        with self.assertRaises(StopAsyncIteration):
            await anext(sink)

    async def test_blocked_producer_wakes_on_cancel(self):
        driver = GatedDriver()
        terminal_chars = []
        sink = CallbackAudioEgress(
            lambda frame: None,
            on_abort=lambda: terminal_chars.append(item.metrics.input_chars),
        )
        item = session(driver, sink, config=SessionConfig(
            delta_queue_size=1, text_queue_size=1, audio_queue_size=1
        ))
        item.start()

        async def produce():
            for _ in range(100):
                await item.push_delta("Sentence.")

        producer = asyncio.create_task(produce())
        await asyncio.wait_for(driver.started.wait(), 1)
        await wait_for_blocked_put(item)
        self.assertFalse(producer.done())
        await asyncio.wait_for(item.cancel(), 1)
        with self.assertRaises(SessionClosedError):
            await asyncio.wait_for(producer, 1)
        self.assertLessEqual(item.metrics.delta_queue_highwater, 1)
        self.assertLessEqual(item.metrics.text_queue_highwater, 1)
        await asyncio.sleep(0)
        self.assertEqual(terminal_chars, [item.metrics.input_chars])
        self.assertTrue(item._delta_queue.empty())
        self.assertTrue(item._commit_queue.empty())
        self.assertTrue(item._audio_queue.empty())
        self.assertEqual(item._pending_puts, set())

    async def test_blocked_producer_wakes_on_failure(self):
        fail = asyncio.Event()

        class LaterFailure(RecordingDriver):
            async def stream(self, request, *, cancel=None):
                self.started.set()
                await fail.wait()
                raise OSError("model crashed")
                yield FRAME

        driver = LaterFailure()
        terminal_chars = []
        sink = CallbackAudioEgress(
            lambda frame: None,
            on_abort=lambda: terminal_chars.append(item.metrics.input_chars),
        )
        item = session(driver, sink, config=SessionConfig(delta_queue_size=1, text_queue_size=1))
        item.start()

        async def produce():
            for _ in range(100):
                await item.push_delta("Sentence.")

        producer = asyncio.create_task(produce())
        await asyncio.wait_for(driver.started.wait(), 1)
        await wait_for_blocked_put(item)
        fail.set()
        with self.assertRaisesRegex(OSError, "model crashed"):
            await asyncio.wait_for(producer, 1)
        with self.assertRaises(OSError):
            await item.wait()
        await asyncio.sleep(0)
        self.assertEqual(terminal_chars, [item.metrics.input_chars])
        self.assertTrue(item._delta_queue.empty())
        self.assertTrue(item._commit_queue.empty())
        self.assertTrue(item._audio_queue.empty())
        self.assertEqual(item._pending_puts, set())

    async def test_finish_caller_cancellation_stops_session(self):
        driver = GatedDriver()
        item = session(driver)
        item.start()
        await item.push_delta("Sentence.")
        await asyncio.wait_for(driver.started.wait(), 1)
        # Hold input admission so finish blocks while finalizing, before wait.
        # Cancelling a task before it ever starts cannot execute its cleanup.
        async with item._input_lock:
            finisher = asyncio.create_task(item.finish())
            await asyncio.sleep(0)
            self.assertFalse(finisher.done())
            finisher.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(finisher, 1)
        self.assertEqual(item.state, SessionState.CANCELLED)

    async def test_finish_wait_cancellation_stops_session(self):
        driver = GatedDriver()
        item = session(driver)
        item.start()
        await item.push_delta("Sentence.", is_final=True)
        await asyncio.wait_for(driver.started.wait(), 1)
        finisher = asyncio.create_task(item.finish())
        await asyncio.sleep(0)  # Enter finish's wait before cancelling it.
        finisher.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(finisher, 1)
        self.assertEqual(item.state, SessionState.CANCELLED)

    async def test_wait_caller_cancellation_keeps_session_alive(self):
        driver = GatedDriver()
        item = session(driver)
        item.start()
        await item.push_delta("Sentence.", is_final=True)
        waiter = asyncio.create_task(item.wait())
        await driver.started.wait()
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertFalse(item.cancel_token.cancelled)
        driver.release.set()
        self.assertEqual((await item.wait()).state, "done")

    async def test_input_limits_do_not_corrupt_valid_input(self):
        item = session(config=SessionConfig(max_delta_chars=8, max_text_chars=10))
        item.start()
        with self.assertRaises(SessionLimitError):
            await item.push_delta("x" * 9)
        await item.push_delta("hello")
        with self.assertRaises(SessionLimitError):
            await item.push_delta("world!")
        await item.push_delta("there")
        self.assertEqual((await item.finish()).input_chars, 10)

    async def test_generation_limit_stops_backend(self):
        driver = RecordingDriver(frames=10)
        item = session(driver, config=SessionConfig(max_audio_seconds=.03))
        item.start()
        await item.push_delta("Sentence.")
        with self.assertRaises(SessionLimitError):
            await asyncio.wait_for(item.finish(), 1)
        self.assertEqual(driver.closed_streams, 1)

    async def test_stalled_synthesis_has_a_deadline(self):
        driver = GatedDriver()
        item = session(driver, config=SessionConfig(synthesis_timeout_s=.02))
        item.start()
        await item.push_delta("Sentence.")
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(item.finish(), 1)
        self.assertEqual(driver.closed_streams, 1)
        self.assertEqual(item.state, SessionState.FAILED)

    async def test_backend_self_cancellation_does_not_hang_siblings(self):
        class SelfCancellingDriver(RecordingDriver):
            async def stream(self, request, *, cancel=None):
                raise asyncio.CancelledError
                yield FRAME

        item = session(SelfCancellingDriver())
        item.start()
        await item.push_delta("Sentence.")
        with self.assertRaisesRegex(RuntimeError, "cancelled unexpectedly"):
            await asyncio.wait_for(item.finish(), 1)

    async def test_format_change_fails_explicitly(self):
        class ChangingDriver(RecordingDriver):
            async def stream(self, request, *, cancel=None):
                yield FRAME
                yield AudioFrame(b"\x00\x00" * 480, AudioFormat(24_000))

        item = session(ChangingDriver())
        item.start()
        await item.push_delta("Sentence.")
        with self.assertRaisesRegex(ValueError, "changed audio format"):
            await item.finish()


class ManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_abort_callback_can_stop_another_session(self):
        async with SpeakTool(RecordingDriver(), egress_factory=NullAudioEgress) as tool:
            stopped = []

            async def on_abort():
                stopped.append(await tool.stop("B"))

            first = await tool.start("A", egress=CallbackAudioEgress(lambda frame: None, on_abort=on_abort))
            second = await tool.start("B")
            await asyncio.wait_for(tool.stop("A"), 1)
            self.assertEqual(first.state, SessionState.CANCELLED)
            self.assertEqual(second.state, SessionState.CANCELLED)
            self.assertEqual(len(stopped), 1)
            self.assertEqual(tool.sessions, {})

    async def test_replacement_abort_callback_can_stop_another_session(self):
        async with SpeakTool(RecordingDriver(), egress_factory=NullAudioEgress) as tool:
            async def on_abort():
                await tool.stop("B")

            original = await tool.start("A", egress=CallbackAudioEgress(lambda frame: None, on_abort=on_abort))
            second = await tool.start("B")
            replacement = await asyncio.wait_for(tool.start("A"), 1)
            self.assertIs(tool.sessions["A"], replacement)
            self.assertEqual(original.state, SessionState.CANCELLED)
            self.assertEqual(second.state, SessionState.CANCELLED)

    async def test_same_id_reentrant_start_supersedes_reservation_without_deadlock(self):
        async with SpeakTool(RecordingDriver(), egress_factory=NullAudioEgress) as tool:
            nested = []

            async def on_abort():
                nested.append(await tool.start("same"))

            original = await tool.start("same", egress=CallbackAudioEgress(lambda frame: None, on_abort=on_abort))
            with self.assertRaisesRegex(SessionClosedError, "superseded"):
                await asyncio.wait_for(tool.start("same"), 1)
            self.assertEqual(original.state, SessionState.CANCELLED)
            self.assertEqual(len(nested), 1)
            self.assertIs(tool.sessions["same"], nested[0])
            self.assertEqual(nested[0].state, SessionState.BUFFERING)

    async def test_last_concurrent_start_wins_without_resurrecting_older_one(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def on_abort():
            entered.set()
            await release.wait()

        async with SpeakTool(RecordingDriver(), egress_factory=NullAudioEgress) as tool:
            await tool.start("same", egress=CallbackAudioEgress(lambda frame: None, on_abort=on_abort))
            older = asyncio.create_task(tool.start("same"))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                reserved = tool.sessions["same"]
                newest = await asyncio.wait_for(tool.start("same"), 1)
                self.assertIs(tool.sessions["same"], newest)
                self.assertEqual(reserved.state, SessionState.CANCELLED)
                release.set()
                with self.assertRaisesRegex(SessionClosedError, "superseded"):
                    await asyncio.wait_for(older, 1)
                self.assertIs(tool.sessions["same"], newest)
            finally:
                release.set()
                await asyncio.gather(older, return_exceptions=True)

    async def test_close_waits_for_detached_stop_cleanup(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def on_abort():
            entered.set()
            await release.wait()

        driver = RecordingDriver()
        tool = SpeakTool(driver, egress_factory=NullAudioEgress)
        await tool.start("A", egress=CallbackAudioEgress(lambda frame: None, on_abort=on_abort))
        stopping = asyncio.create_task(tool.stop("A"))
        closing = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            closing = asyncio.create_task(tool.aclose())
            await asyncio.sleep(.01)
            self.assertFalse(closing.done())
            self.assertFalse(driver.closed)
            release.set()
            await asyncio.wait_for(asyncio.gather(stopping, closing), 1)
            self.assertTrue(driver.closed)
        finally:
            release.set()
            tasks = [stopping] + ([closing] if closing is not None else [])
            await asyncio.gather(*tasks, return_exceptions=True)
            await tool.aclose()

    async def test_close_during_replacement_cannot_install_new_session(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def on_abort():
            entered.set()
            await release.wait()

        driver = RecordingDriver()
        tool = SpeakTool(driver, egress_factory=NullAudioEgress)
        await tool.start("same", egress=CallbackAudioEgress(lambda frame: None, on_abort=on_abort))
        starting = asyncio.create_task(tool.start("same"))
        closing = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            reserved = tool.sessions["same"]
            closing = asyncio.create_task(tool.aclose())
            await asyncio.sleep(.01)
            self.assertFalse(closing.done())
            release.set()
            with self.assertRaisesRegex(RuntimeError, "closed during"):
                await asyncio.wait_for(starting, 1)
            await asyncio.wait_for(closing, 1)
            self.assertEqual(reserved.state, SessionState.CANCELLED)
            self.assertEqual(tool.sessions, {})
            self.assertTrue(driver.closed)
        finally:
            release.set()
            tasks = [starting] + ([closing] if closing is not None else [])
            await asyncio.gather(*tasks, return_exceptions=True)
            await tool.aclose()

    async def test_sessions_do_not_share_cancellation(self):
        driver = RecordingDriver(frames=20, delay=.002)
        async with SpeakTool(driver, CHUNKER, NullAudioEgress) as tool:
            first = await tool.start("first")
            second = await tool.start("second")
            await tool.delta("first", "First.")
            await tool.delta("second", "Second.")
            await driver.started.wait()
            await tool.stop("first")
            metrics = await tool.end("second")
            self.assertTrue(first.cancel_token.cancelled)
            self.assertFalse(second.cancel_token.cancelled)
            self.assertEqual(metrics.state, "done")
            self.assertEqual(metrics.frames, 20)
        self.assertTrue(driver.closed)

    async def test_replacement_survives_old_end_cleanup(self):
        driver = GatedDriver()
        async with SpeakTool(driver, CHUNKER, NullAudioEgress) as tool:
            original = await tool.start("same")
            await tool.delta("same", "First.", is_final=True)
            ending = asyncio.create_task(tool.end("same"))
            await driver.started.wait()
            await asyncio.sleep(0)  # Let end() bind to the original generation.
            replacement = await tool.start("same")
            await ending
            self.assertIs(tool.sessions["same"], replacement)
            self.assertEqual(original.state, SessionState.CANCELLED)
            await tool.stop("same")

    async def test_manager_bounds_sessions_and_stops_idempotently(self):
        driver = RecordingDriver()
        async with SpeakTool(driver, egress_factory=NullAudioEgress, max_sessions=1) as tool:
            await tool.start("one")
            with self.assertRaisesRegex(RuntimeError, "maximum active sessions"):
                await tool.start("two")
            await tool.stop("one")
            self.assertIsNone(await tool.stop("one"))
            await tool.start("two")
        self.assertEqual(tool.sessions, {})
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await tool.start("after-close")

    async def test_unknown_delta_is_not_silently_lost(self):
        async with SpeakTool(RecordingDriver(), egress_factory=NullAudioEgress) as tool:
            with self.assertRaises(KeyError):
                await tool.delta("typo", "hello")

    async def test_optional_external_driver_ownership(self):
        driver = RecordingDriver()
        async with SpeakTool(driver, egress_factory=NullAudioEgress, close_driver=False):
            pass
        self.assertFalse(driver.closed)


class ConfigTests(unittest.TestCase):
    def test_invalid_bounds(self):
        for name, value in (
            ("delta_queue_size", 0), ("text_queue_size", True),
            ("max_frame_bytes", -1), ("synthesis_timeout_s", float("nan")),
            ("max_audio_seconds", float("inf")),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    SessionConfig(**{name: value})
