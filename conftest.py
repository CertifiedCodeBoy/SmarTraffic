"""Pytest configuration and shared fixtures."""
import sys
from pathlib import Path

# Make project root importable during tests
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
