import struct
import unittest

from audri.wav import WavFormatError, WavStreamParser


def wav_bytes(pcm, *, rate=44100, channels=1, streaming=False, extra=b""):
    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * channels * 2, channels * 2, 16)
    body = b"WAVE" + extra + b"fmt " + struct.pack("<I", 16) + fmt
    body += b"data" + struct.pack("<I", 0xFFFFFFFF if streaming else len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", 0xFFFFFFFF if streaming else len(body)) + body


class WavParserTests(unittest.TestCase):
    def test_every_small_transport_chunk_size_preserves_audio(self):
        pcm = struct.pack("<" + "h" * 90, *range(-45, 45))
        for streaming in (False, True):
            for chunk_size in range(1, 55):
                with self.subTest(streaming=streaming, chunk_size=chunk_size):
                    wire = wav_bytes(pcm, channels=2, rate=48000, streaming=streaming)
                    parser = WavStreamParser()
                    output = b"".join(
                        parser.feed(wire[i:i + chunk_size])
                        for i in range(0, len(wire), chunk_size)
                    )
                    parser.finish()
                    self.assertEqual(output, pcm)
                    self.assertEqual(parser.format.sample_rate, 48000)
                    self.assertEqual(parser.format.channels, 2)

    def test_odd_sized_ancillary_chunk_padding(self):
        junk = b"JUNK" + struct.pack("<I", 3) + b"abc\x00"
        wire = wav_bytes(b"\x00\x80\xff\x7f", extra=junk)
        parser = WavStreamParser()
        self.assertEqual(b"".join(parser.feed(bytes([b])) for b in wire), b"\x00\x80\xff\x7f")
        parser.finish()

    def test_finite_truncation_is_not_success(self):
        parser = WavStreamParser()
        parser.feed(wav_bytes(b"\x00\x00" * 100)[:-2])
        with self.assertRaisesRegex(WavFormatError, "Truncated"):
            parser.finish()

    def test_unknown_stream_cannot_end_with_half_a_sample(self):
        parser = WavStreamParser()
        parser.feed(wav_bytes(b"\x00\x00\x00", streaming=True))
        with self.assertRaisesRegex(WavFormatError, "sample frame"):
            parser.finish()

    def test_stereo_stream_cannot_end_between_channels(self):
        parser = WavStreamParser()
        parser.feed(wav_bytes(b"\x00\x00", channels=2, streaming=True))
        with self.assertRaises(WavFormatError):
            parser.finish()

    def test_compressed_and_float_are_rejected(self):
        for tag in (3, 6, 0xFFFE):
            wire = bytearray(wav_bytes(b"\0\0"))
            struct.pack_into("<H", wire, 20, tag)
            with self.subTest(tag=tag), self.assertRaisesRegex(WavFormatError, "PCM16"):
                WavStreamParser().feed(bytes(wire))

    def test_bad_block_alignment_is_rejected(self):
        wire = bytearray(wav_bytes(b"\0\0"))
        struct.pack_into("<H", wire, 32, 4)
        with self.assertRaisesRegex(WavFormatError, "alignment"):
            WavStreamParser().feed(bytes(wire))

    def test_metadata_budget_checked_before_reading_chunk(self):
        wire = b"RIFF\xff\xff\xff\xffWAVEJUNK" + struct.pack("<I", 1 << 30)
        with self.assertRaisesRegex(WavFormatError, "limit"):
            WavStreamParser(max_header_bytes=1024).feed(wire)

    def test_data_before_format_and_non_wave_rejected(self):
        for wire in (b"RIFF\xff\xff\xff\xffWAVEdata\x02\0\0\0\0\0", b"not a WAV!!!"):
            with self.subTest(wire=wire), self.assertRaises(WavFormatError):
                WavStreamParser().feed(wire)

    def test_no_audio_and_truncated_header_are_rejected(self):
        for wire in (b"", b"RIFF", wav_bytes(b"")):
            parser = WavStreamParser()
            parser.feed(wire)
            with self.subTest(wire=wire), self.assertRaises(WavFormatError):
                parser.finish()

    def test_declared_container_length_is_enforced(self):
        with self.assertRaisesRegex(WavFormatError, "RIFF length"):
            WavStreamParser().feed(wav_bytes(b"\0\0") + b"garbage")

    def test_finish_is_idempotent_and_feed_after_finish_fails(self):
        parser = WavStreamParser()
        parser.feed(wav_bytes(b"\0\0"))
        parser.finish()
        parser.finish()
        with self.assertRaises(WavFormatError):
            parser.feed(b"")


if __name__ == "__main__":
    unittest.main()
