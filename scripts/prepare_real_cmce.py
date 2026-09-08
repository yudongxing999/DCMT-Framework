#!/usr/bin/env python3
"""Real CMCE prep — loads companion prepare_real.payload.b64 (gzip)."""
import gzip, base64
from pathlib import Path
_payload = Path(__file__).with_name("prepare_real.payload.b64").read_text().strip()
_CODE = gzip.decompress(base64.b64decode(_payload)).decode()
exec(compile(_CODE, str(Path(__file__).resolve()), "exec"))
