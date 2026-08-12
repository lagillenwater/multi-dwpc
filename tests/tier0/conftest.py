import sys
from pathlib import Path

# Add src directory to Python path so modules can be imported directly
# (from tests/tier0/conftest.py, go up two levels to repo root, then into src)
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
