#!/usr/bin/env python3
"""Load prepare_real.part*.b64 / payload.b64 (CRC32), raise n_total cap to 500 in-memory."""
import gzip, base64, zlib, re
from pathlib import Path
_here = Path(__file__).resolve().parent
_parts = sorted(_here.glob("prepare_real.part*.b64"), key=lambda p: int(p.stem.split("part")[-1]))
if _parts:
    _b64 = "".join(p.read_text().strip() for p in _parts)
else:
    _mono = _here / "prepare_real.payload.b64"
    if not _mono.exists():
        raise FileNotFoundError("missing prepare_real.part*.b64 / prepare_real.payload.b64")
    _b64 = _mono.read_text().strip()
_raw = gzip.decompress(base64.b64decode(_b64))
_expect_crc = 3883616231
_got = zlib.crc32(_raw) & 0xffffffff
if _got != _expect_crc:
    raise RuntimeError(f"prepare_real payload CRC mismatch: {_got:08x} != {_expect_crc:08x}")
_src = _raw.decode()
_src2, _n = re.subn(
    r"n = min\(max\(args\.n_total, 1\), 120\)",
    "n = min(max(args.n_total, 1), 500)  # raised for scale experiments (was 120)",
    _src,
    count=1,
)
if _n != 1:
    raise RuntimeError(f"failed to patch n_total cap (replacements={_n})")
exec(compile(_src2, str(Path(__file__).resolve()), "exec"))
