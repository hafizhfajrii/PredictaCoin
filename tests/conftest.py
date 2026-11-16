"""Konfigurasi pytest: tambahkan folder skrip ke jalur impor."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/data", "src/models"):
    sys.path.insert(0, str(ROOT / sub))
