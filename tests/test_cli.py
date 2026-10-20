import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from audri.cli import main


class CLITests(unittest.TestCase):
    def invoke(self, args, stdin=""):
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors), patch("sys.stdin", io.StringIO(stdin)):
            code = main(args)
        return code, output.getvalue(), errors.getvalue()

    def test_tone_speak_writes_real_wav_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.wav"
            code, output, errors = self.invoke(["speak", "Hello world.", "--backend", "tone", "--output", str(target)])
            self.assertEqual(code, 0, errors)
            self.assertIn("not speech", errors)
            metrics = json.loads(output)
            with wave.open(str(target), "rb") as source:
                self.assertEqual(source.getframerate(), metrics["sample_rate"])
                self.assertEqual(source.getnchannels(), 1)
                self.assertGreater(source.getnframes(), 0)
                self.assertEqual(len(source.readframes(source.getnframes())), metrics["bytes"])
            self.assertGreater(metrics["audio_seconds"], 0)

    def test_stream_preserves_pipeline_end_and_writes_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "stream.wav"
            code, output, errors = self.invoke(["stream", "--backend", "tone", "--output", str(target)],
                                               "First sentence.\n第二个句子。\n")
            self.assertEqual(code, 0, errors)
            metrics = json.loads(output)
            self.assertEqual(metrics["input_chars"], len("First sentence.\n第二个句子。\n"))
            with wave.open(str(target), "rb") as source:
                self.assertGreater(source.getnframes(), 0)

    def test_speak_reads_stdin(self):
        with tempfile.TemporaryDirectory() as directory:
            code, output, errors = self.invoke(["speak", "--stdin", "--backend", "tone", "--output", str(Path(directory) / "a.wav")], "Read me.")
            self.assertEqual(code, 0, errors)
            self.assertGreater(json.loads(output)["frames"], 0)

    def test_missing_or_conflicting_text_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "a.wav"
            for args in (["speak"], ["speak", "hello", "--stdin"]):
                code, _, errors = self.invoke([*args, "--backend", "tone", "--output", str(target)])
                self.assertEqual(code, 1)
                self.assertIn("exactly one", errors)
                self.assertFalse(target.exists())

    def test_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "a.wav"
            target.write_bytes(b"keep this file")
            code, _, _ = self.invoke(["speak", "hello", "--backend", "tone", "--output", str(target)])
            self.assertEqual(code, 1)
            self.assertEqual(target.read_bytes(), b"keep this file")

    def test_blank_input_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            code, _, errors = self.invoke(["speak", " ", "--backend", "tone", "--output", str(Path(directory) / "a.wav")])
            self.assertEqual(code, 1)
            self.assertIn("blank", errors)

    def test_benchmark_exposes_each_measurement_and_labels_diagnostic(self):
        code, output, errors = self.invoke(["benchmark", "hello", "--backend", "tone", "--runs", "2"])
        self.assertEqual(code, 0, errors)
        metrics = json.loads(output)
        self.assertTrue(metrics["diagnostic_only"])
        self.assertEqual(len(metrics["runs"]), 2)
        self.assertGreaterEqual(metrics["first_audio_p95_ms"], metrics["first_audio_p50_ms"])
        self.assertGreaterEqual(metrics["rtf_p95"], metrics["rtf_p50"])

    def test_reference_audio_requires_transcript_before_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "ref.wav"
            reference.write_bytes(b"a")
            code, _, errors = self.invoke(["speak", "hello", "--backend", "tone", "--reference-audio", str(reference),
                                           "--output", str(Path(directory) / "a.wav")])
            self.assertEqual(code, 1)
            self.assertIn("--reference-text", errors)


if __name__ == "__main__":
    unittest.main()
