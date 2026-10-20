import asyncio
import json
import subprocess
import sys
import unittest

from audri.audio import AudioFormat, AudioFrame, PCMFramer
from audri.tts.chunker import ChunkerConfig, TextChunker
from audri.tts.mock import ToneDriver
from audri.tts.types import CancellationToken, SynthesisRequest, VoiceReference


class AudioTests(unittest.TestCase):
    def test_44100_framing_keeps_every_sample(self):
        format = AudioFormat(44_100)
        pcm = bytes(range(256)) * 17
        framer = PCMFramer(format)
        frames = []
        for offset in range(0, len(pcm), 13):
            frames.extend(framer.feed(pcm[offset:offset + 13]))
        frames.extend(framer.finish())
        self.assertEqual(b"".join(f.pcm for f in frames), pcm)
        self.assertEqual(frames[0].samples, 882)
        self.assertEqual(frames[0].duration_s, .02)
        self.assertEqual(sum(f.samples for f in frames), len(pcm) // 2)
        self.assertLess(frames[-1].samples, 882)

    def test_invalid_audio_fails_instead_of_playing_noise(self):
        for kwargs in ({"sample_rate": 0}, {"sample_rate": True},
                       {"sample_rate": 24_000, "channels": 0},
                       {"sample_rate": 24_000, "sample_width": 4}):
            with self.assertRaises(ValueError):
                AudioFormat(**kwargs)
        with self.assertRaises(ValueError):
            AudioFrame(b"a", AudioFormat(24_000))
        framer = PCMFramer(AudioFormat(24_000))
        framer.feed(b"a")
        with self.assertRaises(ValueError):
            framer.finish()

    def test_stereo_alignment(self):
        format = AudioFormat(48_000, channels=2)
        self.assertEqual(AudioFrame(b"\0" * 3840, format).duration_s, .02)
        with self.assertRaises(ValueError):
            AudioFrame(b"\0\0", format)

    def test_imports_do_not_load_optional_model_stack(self):
        code = (
            "import audri, audri.tts.driver, audri.tts.fish, audri.tts.qwen; "
            "import json, sys; print(json.dumps(sorted(set(sys.modules) & "
            "{'torch', 'qwen_tts', 'numpy', 'httpx', 'sounddevice', 'fastapi'})))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [])


class ChunkerTests(unittest.TestCase):
    def collect(self, text, sizes=(1, 2, 7, 31, 503), config=None):
        chunker = TextChunker(config)
        result = []
        index = 0
        turn = 0
        while index < len(text):
            size = sizes[turn % len(sizes)]
            ready = chunker.push(text[index:index + size])
            while ready is not None:
                result.append(ready)
                ready = chunker.pop_ready()
            index += size
            turn += 1
        while (last := chunker.flush()) is not None:
            result.append(last)
        return result

    def test_long_delta_and_final_flush_are_bounded(self):
        text = ("A sentence with useful details. " * 30) + ("龙" * 401)
        result = self.collect(text, sizes=(100_000,))
        self.assertTrue(all(0 < len(chunk) <= 180 for chunk in result))
        self.assertEqual("".join("".join(result).split()), "".join(text.split()))

    def test_chunking_does_not_change_non_whitespace_text(self):
        for text in ["Hello. " * 99, "这是中文语音。这个系统支持中文！" * 50,
                     "abc" * 300, "[whisper] hello [speaker_0] again. " * 30,
                     "Dr. Yang measured 3.14 milliseconds. " * 30]:
            with self.subTest(text=text[:50]):
                chunks = self.collect(text)
                self.assertEqual("".join("".join(chunks).split()), "".join(text.split()))
                self.assertTrue(all(len(chunk) <= 180 for chunk in chunks))

    def test_cjk_sentence_boundary(self):
        chunker = TextChunker(ChunkerConfig(min_chars_to_start=3, max_chars_per_chunk=20))
        self.assertEqual(chunker.push("这是中文。下一句"), "这是中文。")
        self.assertEqual(chunker.flush(), "下一句")

    def test_partial_control_is_not_split_on_timeout(self):
        chunker = TextChunker(ChunkerConfig(min_chars_to_start=3, max_chars_per_chunk=20))
        self.assertIsNone(chunker.push("[whis"))
        chunker.last_delta_ts -= 1
        self.assertIsNone(chunker.check_timeout())
        self.assertEqual(chunker.push("per] Hello there!"), "[whisper] Hello")
        self.assertEqual(chunker.flush(), "there!")

    def test_abbreviation_and_decimal_are_not_sentence_boundaries(self):
        chunker = TextChunker(ChunkerConfig(min_chars_to_start=1, max_chars_per_chunk=80))
        self.assertIsNone(chunker.push("Dr. Yang measured 3.14"))
        self.assertEqual(chunker.push(" seconds."), "Dr. Yang measured 3.14 seconds.")

    def test_control_opener_split_between_deltas(self):
        config = ChunkerConfig(min_chars_to_start=3, max_chars_per_chunk=20)
        chunker = TextChunker(config)
        self.assertEqual(chunker.push("x" * 19 + "<"), "x" * 19)
        chunker.last_delta_ts -= 1
        self.assertIsNone(chunker.check_timeout())
        self.assertIsNone(chunker.push("|speaker:0|>"))
        self.assertEqual(chunker.flush(), "<|speaker:0|>")

    def test_invalid_config_and_overlong_control(self):
        for kwargs in ({"max_chars_per_chunk": 0}, {"min_chars_to_start": 200},
                       {"boundary_timeout_ms": float("nan")}):
            with self.assertRaises(ValueError):
                ChunkerConfig(**kwargs)
        with self.assertRaises(ValueError):
            TextChunker().push("[" + "x" * 300 + "]")


class RequestTests(unittest.TestCase):
    def test_frozen_request_and_reference_validation(self):
        for kwargs in ({"top_p": float("nan")}, {"seed": -1}, {"max_new_tokens": True},
                       {"references": []}, {"normalize": "yes"},
                       {"reference_id": "voice", "references": (VoiceReference(b"a", "text"),)}):
            with self.assertRaises((ValueError, TypeError)):
                SynthesisRequest("Hello", **kwargs)
        self.assertEqual(VoiceReference(b"abc", "sample", 3).as_dict()["audio"], "YWJj")
        self.assertEqual(SynthesisRequest("").text, "")


class DiagnosticTests(unittest.IsolatedAsyncioTestCase):
    async def test_tone_has_exact_unpadded_duration(self):
        driver = ToneDriver(sample_rate=44_100, duration_s=.123)
        frames = [frame async for frame in driver.stream(SynthesisRequest("Test"))]
        self.assertEqual(sum(f.samples for f in frames), round(44_100 * .123))
        self.assertEqual(frames[0].format.sample_rate, 44_100)
        self.assertEqual([f async for f in driver.stream(SynthesisRequest("  "))], [])

    async def test_cancellation_is_scoped(self):
        driver = ToneDriver()
        cancelled = CancellationToken()
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await anext(driver.stream(SynthesisRequest("A"), cancel=cancelled))
        self.assertTrue([f async for f in driver.stream(SynthesisRequest("B"))])
