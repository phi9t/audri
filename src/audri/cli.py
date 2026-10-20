"""Command line entry points; model and service dependencies load only on demand."""

from __future__ import annotations

import argparse
import asyncio
import codecs
import io
import json
import math
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from .tts.types import SynthesisRequest, VoiceReference


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("fish", "qwen", "tone"), default="fish")
    parser.add_argument("--url", default=os.environ.get("FISH_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--timeout", type=_positive, default=120.0)
    parser.add_argument("--model", help="Qwen model ID; Fish chooses its model at server startup")


def _voice_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--voice", default="default")
    parser.add_argument("--language", default="English")
    parser.add_argument("--instruction")
    parser.add_argument("--reference-id", help="Named reference on the Fish server")
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text", help="Transcript corresponding to --reference-audio")
    parser.add_argument("--seed", type=int)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audri", description="Streaming speech with explicit audio formats.")
    sub = parser.add_subparsers(dest="command", required=True)

    speak = sub.add_parser("speak", help="Synthesize text into a finalized WAV file")
    speak.add_argument("text", nargs="?")
    speak.add_argument("--stdin", action="store_true", help="Read complete text from stdin")
    speak.add_argument("--output", type=Path, required=True)
    speak.add_argument("--overwrite", action="store_true")
    _backend_arguments(speak)
    _voice_arguments(speak)

    stream = sub.add_parser("stream", help="Stream UTF-8 stdin as text arrives into a WAV file")
    stream.add_argument("--output", type=Path, required=True)
    stream.add_argument("--overwrite", action="store_true")
    _backend_arguments(stream)
    _voice_arguments(stream)

    doctor = sub.add_parser("doctor", help="Check a running Fish Speech service")
    doctor.add_argument("--url", default=os.environ.get("FISH_URL", "http://127.0.0.1:8080"))
    doctor.add_argument("--timeout", type=_positive, default=10.0)

    benchmark = sub.add_parser("benchmark", help="Measure client-observed latency and real-time factor")
    benchmark.add_argument("text")
    benchmark.add_argument("--runs", type=_positive_int, default=3)
    _backend_arguments(benchmark)
    _voice_arguments(benchmark)

    serve = sub.add_parser("serve", help="Serve HTTP speech and a token-stream WebSocket")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_positive_int, default=8060)
    serve.add_argument("--max-sessions", type=_positive_int, default=8)
    _backend_arguments(serve)
    return parser


def _make_driver(args: argparse.Namespace):
    if args.backend == "fish":
        if args.model:
            raise ValueError("--model is only for Qwen; select Fish checkpoints on the Fish server")
        from .tts.fish import FishSpeechDriver

        return FishSpeechDriver(base_url=args.url, timeout=args.timeout, api_key=os.environ.get("FISH_API_KEY"))
    if args.backend == "qwen":
        from .tts.driver import Qwen3TTSDriver

        return Qwen3TTSDriver(**({"model_name": args.model} if args.model else {}))
    if args.model:
        raise ValueError("--model is only for Qwen")
    from .tts.mock import ToneDriver

    print("Diagnostic tone backend: generated audio is a test signal, not speech.", file=sys.stderr)
    return ToneDriver()


def _request(args: argparse.Namespace, text: str) -> SynthesisRequest:
    references: tuple[VoiceReference, ...] = ()
    if args.reference_audio:
        if args.reference_id:
            raise ValueError("choose --reference-id or --reference-audio")
        if args.reference_text is None:
            raise ValueError("--reference-audio requires --reference-text")
        if args.reference_audio.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("reference audio exceeds 8 MiB")
        references = (VoiceReference(audio=args.reference_audio.read_bytes(), text=args.reference_text),)
    elif args.reference_text is not None:
        raise ValueError("--reference-text requires --reference-audio")
    return SynthesisRequest(
        text=text,
        voice=args.voice,
        language=args.language,
        instruction=args.instruction,
        reference_id=args.reference_id,
        references=references,
        seed=args.seed,
    )


async def _stdin_deltas(stream: TextIO):
    """Read a POSIX pipe/terminal without an uncancellable executor thread.

    Partial UTF-8 characters are retained until their remaining bytes arrive.
    Input is delivered as available, without requiring a newline or full block.
    """
    if isinstance(stream, io.StringIO):
        while delta := stream.read(1024):
            yield delta
            await asyncio.sleep(0)
        return
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError, io.UnsupportedOperation) as exc:
        raise ValueError("stream input must be a file, POSIX pipe, or terminal") from exc
    regular_file = stat.S_ISREG(os.fstat(descriptor).st_mode)
    if os.name != "posix" and not regular_file:
        raise RuntimeError("asynchronous stdin streaming currently requires a POSIX system")
    loop = asyncio.get_running_loop()
    decoder = codecs.getincrementaldecoder("utf-8")()
    while True:
        if regular_file:
            chunk = os.read(descriptor, 4096)
        else:
            ready = loop.create_future()

            def readable():
                if not ready.done():
                    ready.set_result(None)

            loop.add_reader(descriptor, readable)
            try:
                await ready
                chunk = os.read(descriptor, 4096)
            finally:
                loop.remove_reader(descriptor)
        text = decoder.decode(chunk, final=not chunk)
        if text:
            yield text
        if not chunk:
            return
        await asyncio.sleep(0)


async def _consume(driver, request: SynthesisRequest, sink=None) -> dict[str, Any]:
    started = time.monotonic()
    first: float | None = None
    duration = 0.0
    count = 0
    byte_count = 0
    audio_format = None
    iterator = driver.stream(request)
    try:
        async for frame in iterator:
            if first is None:
                first = (time.monotonic() - started) * 1000
                audio_format = frame.format
            elif frame.format != audio_format:
                raise ValueError("backend changed audio format within one request")
            if sink is not None:
                await sink.write(frame)
            duration += frame.duration_s
            count += 1
            byte_count += len(frame.pcm)
            if duration > 3600:
                raise ValueError("a single synthesis exceeds the one-hour audio limit")
            await asyncio.sleep(0)
        if count == 0:
            raise RuntimeError("backend returned no audio")
        if sink is not None:
            await sink.close()
    except BaseException:
        if sink is not None:
            await sink.abort()
        raise
    finally:
        await iterator.aclose()
    elapsed = time.monotonic() - started
    return {
        "first_audio_ms": first,
        "elapsed_seconds": elapsed,
        "audio_seconds": duration,
        "rtf": elapsed / duration if duration else None,
        "frames": count,
        "bytes": byte_count,
        "sample_rate": audio_format.sample_rate if audio_format else None,
        "channels": audio_format.channels if audio_format else None,
    }


async def _execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "doctor":
        from .tts.fish import FishSpeechDriver

        driver = FishSpeechDriver(base_url=args.url, timeout=args.timeout, api_key=os.environ.get("FISH_API_KEY"))
        try:
            result = {"url": args.url, "health": await driver.health()}
            if hasattr(driver, "defaults"):
                result["defaults"] = await driver.defaults()
            return result
        finally:
            await driver.aclose()

    driver = _make_driver(args)
    try:
        if args.command == "speak":
            from .tts.audio_egress import WaveAudioEgress

            if args.stdin == (args.text is not None):
                raise ValueError("provide text or --stdin, exactly one")
            text = sys.stdin.read(100001) if args.stdin else args.text
            if not text or not text.strip():
                raise ValueError("text must not be blank")
            if len(text) > 100000:
                raise ValueError("text exceeds 100000 characters; use stream for incremental input")
            sink = WaveAudioEgress(args.output, overwrite=args.overwrite)
            async with asyncio.timeout(args.timeout):
                result = await _consume(driver, _request(args, text), sink)
            return {"output": str(args.output), "backend": args.backend, **result}

        if args.command == "stream":
            from .speak_tool import SpeakTool
            from .tts.audio_egress import WaveAudioEgress
            from .tts.session import SessionConfig

            sink = WaveAudioEgress(args.output, overwrite=args.overwrite)
            async with SpeakTool(driver, close_driver=False, session_config=SessionConfig(synthesis_timeout_s=args.timeout)) as tool:
                await tool.start("cli", request=_request(args, ""), egress=sink)
                async for delta in _stdin_deltas(sys.stdin):
                    await tool.delta("cli", delta)
                metrics = await tool.end("cli")
                return {"output": str(args.output), "backend": args.backend, **metrics.as_dict()}

        if args.command == "benchmark":
            if args.runs > 100:
                raise ValueError("--runs must be at most 100")
            request = _request(args, args.text)
            if not request.text.strip():
                raise ValueError("text must not be blank")
            runs = []
            for _ in range(args.runs):
                async with asyncio.timeout(args.timeout):
                    runs.append(await _consume(driver, request))
            latencies = sorted(run["first_audio_ms"] for run in runs)
            rtfs = sorted(run["rtf"] for run in runs)

            def percentile(values, fraction):
                return values[max(0, math.ceil(len(values) * fraction) - 1)]

            return {
                "backend": args.backend,
                "measurement": "Client wall-clock, including transport and backpressure; first run may include warmup.",
                "diagnostic_only": args.backend == "tone",
                "runs": runs,
                "first_audio_p50_ms": percentile(latencies, 0.5),
                "first_audio_p95_ms": percentile(latencies, 0.95),
                "rtf_p50": percentile(rtfs, 0.5),
                "rtf_p95": percentile(rtfs, 0.95),
            }
        raise ValueError(f"unsupported command: {args.command}")
    finally:
        await driver.aclose()


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "serve":
            if args.port > 65535:
                raise ValueError("--port must be at most 65535")
            try:
                import uvicorn
                from .server import create_app
            except ImportError as exc:
                raise RuntimeError("The serve command requires Audri's server dependencies") from exc
            app = create_app(_make_driver(args), api_key=os.environ.get("AUDRI_API_KEY"), max_sessions=args.max_sessions)
            uvicorn.run(app, host=args.host, port=args.port, ws_max_size=4 * 1024 * 1024)
        else:
            print(json.dumps(asyncio.run(_execute(args)), ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"audri: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
