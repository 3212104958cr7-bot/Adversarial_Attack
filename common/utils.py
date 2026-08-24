import json
import logging
import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref="auto"):
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "auto" or pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "cuda":
            logging.warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(pref)


def setup_logger(name, log_path=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def append_json_log(log_path, record):
    """Appends one record to a JSON file holding a list of records,
    creating the file/parent directory if needed.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    records = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []
    records.append(record)
    with open(log_path, "w") as f:
        json.dump(records, f, indent=2)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
