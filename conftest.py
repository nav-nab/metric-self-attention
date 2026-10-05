"""Puts the repository root on sys.path so tests in tests/ and shelved/*/tests/ can import `msa` and `shelved`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
