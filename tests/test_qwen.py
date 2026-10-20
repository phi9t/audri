"""Test native-worker ownership without loading Torch or any checkpoint."""

import asyncio
import threading
import unittest

from audri.tts.qwen import Qwen3TTSDriver
from audri.tts.types import CancellationToken, SynthesisRequest


class QwenWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_call_keeps_model_exclusive_until_worker_finishes(self):
        driver = Qwen3TTSDriver()
        entered = threading.Event()
        release = threading.Event()
        next_entered = threading.Event()

        def first():
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test did not release native worker")
            return 1

        first_task = asyncio.create_task(driver._run_serial(first, None))
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(.005)
            self.assertTrue(entered.is_set())
            first_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first_task
            second = asyncio.create_task(driver._run_serial(lambda: next_entered.set(), None))
            await asyncio.sleep(.02)
            self.assertFalse(next_entered.is_set())
            release.set()
            await asyncio.wait_for(second, 1)
            self.assertTrue(next_entered.is_set())
        finally:
            release.set()
            await driver.aclose()

    async def test_cancel_while_waiting_does_not_steal_model_lock(self):
        driver = Qwen3TTSDriver()
        await driver._lock.acquire()
        token = CancellationToken()
        task = asyncio.create_task(driver._run_serial(lambda: "never", token))
        await asyncio.sleep(0)
        token.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertTrue(driver._lock.locked())
        driver._lock.release()
        self.assertEqual(await driver._run_serial(lambda: "ok", None), "ok")
        await driver.aclose()

    async def test_frame_format_comes_from_model_and_tail_is_unpadded(self):
        class ModelDouble(Qwen3TTSDriver):
            def _synthesize(self, request):
                return b"\0\0" * 1000, 44_100

        async with ModelDouble() as driver:
            frames = [f async for f in driver.stream(SynthesisRequest("Hello"))]
            self.assertEqual([f.samples for f in frames], [882, 118])
            self.assertEqual(frames[0].format.sample_rate, 44_100)

    async def test_closed_driver_cannot_launch_work(self):
        driver = Qwen3TTSDriver()
        await driver.aclose()
        with self.assertRaises(RuntimeError):
            await driver._run_serial(lambda: None, None)
