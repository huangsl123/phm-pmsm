"""Shared paths for scripts moved under the project ``scripts`` directory."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "result_upgrade"
PAPER_ASSETS_DIR = PROJECT_ROOT / "PHM_会议论文_中英文与图"

project_root_text = str(PROJECT_ROOT)
if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)
