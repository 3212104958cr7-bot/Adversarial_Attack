"""Central path resolution so every script works regardless of cwd."""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREDRNET_DIR = os.path.join(PROJECT_ROOT, "models", "predrnet")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

if PREDRNET_DIR not in sys.path:
    sys.path.insert(0, PREDRNET_DIR)
