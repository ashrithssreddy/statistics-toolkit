import sys
from pathlib import Path

import matplotlib


# Keep plotting headless in test runs.
matplotlib.use("Agg")


# Allow direct imports used by utils, e.g. `from ab_utils_01_data_setup import ...`.
AB_TESTING_DIR = Path(__file__).resolve().parents[1]
if str(AB_TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(AB_TESTING_DIR))

