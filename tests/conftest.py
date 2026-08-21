import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Existing executor unit tests deliberately exercise the unmanaged backend seam.
# Production remains fail-closed unless this explicit test-only opt-in is set.
os.environ.setdefault("PLAN_ALLOW_UNMANAGED_TEST_EXECUTION", "1")
