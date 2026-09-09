#!/usr/bin/env python3
"""Load prepare_real parts via *.b64.hex sidecars (preferred) or *.b64; CRC-check; patch n_total to 500."""
import gzip, base64, zlib, re, binascii
from pathlib import Path
_here = Path(__file__).resolve().parent

def _part_idx(p: Path) -> int:
    # prepare_real.partN.b64 or prepare_real.partN.b64.hex
    return int(p.name.split("part")[1].split(".")[0])

_hex_parts = sorted(_here.glob("prepare_real.part*.b64.hex"), key=_part_idx)
if len(_hex_parts) >= 5:
    _b64 = "".join(binascii.unhexlify("".join(p.read_text().split())).decode() for p in _hex_parts)
else:
    _parts = sorted(_here.glob("prepare_real.part*.b64"), key=_part_idx)
    # exclude accidental .b64.hex matched? glob part*.b64 won't match .b64.hex
    if _parts:
        _b64 = "".join(p.read_text().strip() for p in _parts)
    else:
        _mono = _here / "prepare_real.payload.b64"
        if not _mono.exists():
            raise FileNotFoundError("missing prepare_real.part*.b64(.hex) / payload.b64")
        _b64 = _mono.read_text().strip()

_raw = gzip.decompress(base64.b64decode(_b64.strip()))
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
