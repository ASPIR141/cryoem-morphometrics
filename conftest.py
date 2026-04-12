"""Root conftest: add cryoem_cellstate/ to sys.path so all src.* imports resolve."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "cryoem_cellstate"))
