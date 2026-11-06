"""Convenience entrypoint for scripts/compare_two.py."""
import sys
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.compare_two import main

if __name__ == "__main__":
    main()

