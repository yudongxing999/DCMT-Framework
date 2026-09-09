#!/usr/bin/env python3
"""Loads train_tiny.part*.b64 (preferred) or train_tiny.payload.b64 (gzip); verifies CRC32."""
import gzip, base64, zlib
from pathlib import Path
_here = Path(__file__).resolve().parent
_parts = sorted(_here.glob("train_tiny.part*.b64"), key=lambda p: int(p.stem.split("part")[-1]))
if _parts:
    _b64 = "".join(p.read_text().strip() for p in _parts)
else:
    _mono = _here / "train_tiny.payload.b64"
    if not _mono.exists():
        raise FileNotFoundError("missing train_tiny.part*.b64 / train_tiny.payload.b64")
    _b64 = _mono.read_text().strip()
_raw = gzip.decompress(base64.b64decode(_b64))
_expect_crc = 2067398022
_got = zlib.crc32(_raw) & 0xffffffff
if _got != _expect_crc:
    raise RuntimeError(f"train_tiny payload CRC mismatch: {_got:08x} != {_expect_crc:08x}")
exec(compile(_raw.decode(), str(Path(__file__).resolve()), "exec"))
