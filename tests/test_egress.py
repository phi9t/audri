import asyncio
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from audri.audio import AudioFormat, AudioFrame
from audri.tts.audio_egress import (
    CallbackAudioEgress, LocalAudioEgress, NullAudioEgress,
    QueueAudioEgress, WaveAudioEgress,
)


FORMAT = AudioFormat(44_100)
FRAME = AudioFrame(b"\x01\x00" * 882, FORMAT)


class SinkTests(unittest.IsolatedAsyncioTestCase):
    async def test_wav_preserves_native_format_and_exact_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.wav"
            sink = WaveAudioEgress(path)
            await sink.write(FRAME)
            tail = AudioFrame(b"\x02\x00" * 19, FORMAT)
            await sink.write(tail)
            self.assertFalse(path.exists())
            await sink.close()
            with wave.open(str(path), "rb") as wav:
                self.assertEqual(wav.getframerate(), 44_100)
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getnframes(), 901)
                self.assertEqual(wav.readframes(901), FRAME.pcm + tail.pcm)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            await sink.close()

    async def test_wav_abort_does_not_publish_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.wav"
            sink = WaveAudioEgress(path)
            await sink.write(FRAME)
            await sink.abort()
            self.assertTrue(sink.aborted)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    async def test_wav_existing_file_is_never_overwritten_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.wav"
            path.write_bytes(b"existing content")
            sink = WaveAudioEgress(path)
            with self.assertRaises(FileExistsError):
                await sink.write(FRAME)
            await sink.abort()
            self.assertEqual(path.read_bytes(), b"existing content")

    async def test_wav_publish_race_keeps_other_writer_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raced.wav"
            sink = WaveAudioEgress(path)
            await sink.write(FRAME)
            path.write_bytes(b"other writer")
            with self.assertRaises(FileExistsError):
                await sink.close()
            self.assertEqual(path.read_bytes(), b"other writer")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    async def test_wav_explicit_overwrite_and_empty_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.wav"
            path.write_bytes(b"old content")
            sink = WaveAudioEgress(path, overwrite=True, format=AudioFormat(48_000, 2))
            await sink.close()
            with wave.open(str(path), "rb") as wav:
                self.assertEqual(wav.getframerate(), 48_000)
                self.assertEqual(wav.getnchannels(), 2)
                self.assertEqual(wav.getnframes(), 0)

    async def test_sinks_reject_format_changes(self):
        sink = NullAudioEgress()
        await sink.write(FRAME)
        with self.assertRaisesRegex(ValueError, "format changed"):
            await sink.write(AudioFrame(b"\x00\x00", AudioFormat(24_000)))
        await sink.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await sink.write(FRAME)

    async def test_callback_supports_sync_async_and_cleanup(self):
        seen = []
        sync = CallbackAudioEgress(seen.append)
        await sync.write(FRAME)
        await sync.close()
        self.assertEqual(seen, [FRAME])
        events = []

        async def callback(frame):
            await asyncio.sleep(0)
            events.append(frame)

        sink = CallbackAudioEgress(callback, on_close=lambda: events.append("close"))
        await sink.write(FRAME)
        await sink.close()
        await sink.close()
        self.assertEqual(events, [FRAME, "close"])

    async def test_queue_backpressure_and_graceful_close(self):
        sink = QueueAudioEgress(maxsize=1)
        await sink.write(FRAME)
        blocked = asyncio.create_task(sink.write(FRAME))
        await asyncio.sleep(.01)
        self.assertFalse(blocked.done())
        self.assertIs(await anext(sink), FRAME)
        await asyncio.wait_for(blocked, 1)
        await sink.close()
        self.assertIs(await anext(sink), FRAME)
        with self.assertRaises(StopAsyncIteration):
            await anext(sink)

    async def test_queue_abort_wakes_blocked_writer_and_drops_audio(self):
        sink = QueueAudioEgress(maxsize=1)
        await sink.write(FRAME)
        blocked = asyncio.create_task(sink.write(FRAME))
        await asyncio.sleep(.01)
        await sink.abort()
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(blocked, 1)
        self.assertEqual(sink.queue.qsize(), 0)
        with self.assertRaises(StopAsyncIteration):
            await anext(sink)

    async def test_queue_eof_wakes_waiting_reader(self):
        sink = QueueAudioEgress(maxsize=1)
        reader = asyncio.create_task(anext(sink))
        await asyncio.sleep(0)
        await sink.close()
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(reader, 1)

    async def test_queue_reader_cancellation_does_not_lose_next_frame(self):
        sink = QueueAudioEgress(maxsize=1)
        reader = asyncio.create_task(anext(sink))
        await asyncio.sleep(0)
        reader.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await reader
        await sink.write(FRAME)
        self.assertIs(await anext(sink), FRAME)
        await sink.abort()

    async def test_playback_lazily_uses_negotiated_format_and_abort(self):
        fake = MagicMock()
        device = fake.RawOutputStream.return_value
        with patch.dict("sys.modules", {"sounddevice": fake}):
            sink = LocalAudioEgress(device="test-speaker")
            self.assertFalse(fake.RawOutputStream.called)
            await sink.write(FRAME)
            self.assertEqual(fake.RawOutputStream.call_args.kwargs["samplerate"], 44_100)
            device.write.assert_called_once_with(FRAME.pcm)
            await sink.abort()
            device.abort.assert_called_once()
            device.close.assert_called_once()
            device.stop.assert_not_called()
