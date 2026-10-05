"""Puts this folder on sys.path so that tests in tests/ and mqar/tests/ can import `msa` and `mqar`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
