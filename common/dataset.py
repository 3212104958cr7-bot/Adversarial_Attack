"""I-RAVEN dataset wrapper around the PredRNet repo's RAVEN loader.

Re-exposes each RPM problem as (context, candidates, target, config) with
images as float32 tensors scaled to [0, 1], which is the pixel convention
used throughout the attack/ and defense/ code (see coding_agent_prompt.md).
The PredRNet network itself expects images in [-1, 1]; that rescale is done
inside common.model.RPMModel, not here, so [0, 1] stays the single source of
truth for perturbation budgets (epsilon is defined in this space).
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from common.paths import PREDRNET_DIR  # noqa: F401 (ensures sys.path is set)
from data.raven import RAVEN, sub_folders

# Folder name (as used by the I-RAVEN/SRAN generator) -> paper-style config name
FOLDER_TO_CONFIG = {
    "center_single": "Center",
    "distribute_four": "2x2Grid",
    "distribute_nine": "3x3Grid",
    "left_center_single_right_center_single": "Left-Right",
    "up_center_single_down_center_single": "Up-Down",
    "in_center_single_out_center_single": "Out-InCenter",
    "in_distribute_four_out_center_single": "Out-InGrid",
}
CONFIG_NAMES = list(FOLDER_TO_CONFIG.values())


class IRavenDataset(Dataset):
    """Wraps PredRNet's RAVEN dataset, splitting the 16 panels into
    8 context images + 8 answer candidates, and returning pixels in [0, 1].
    """

    def __init__(self, root, split, image_size=80, subset="None"):
        assert split in ("train", "val", "test")
        self.base = RAVEN(
            root, data_split=split, image_size=image_size,
            transform=None, subset=subset,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, target, meta_target, structure_encoded, data_file = self.base[idx]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        image = image.float() / 255.0  # raw pixels are stored in [0, 255]

        context = image[:8]      # (8, H, W)
        candidates = image[8:16]  # (8, H, W)

        config_folder = data_file.split(os.sep)[0]
        config = FOLDER_TO_CONFIG.get(config_folder, config_folder)

        target = int(target)
        return context, candidates, target, config


def make_loader(root, split, image_size=80, batch_size=32, num_workers=4, shuffle=None):
    dataset = IRavenDataset(root, split, image_size=image_size)
    if shuffle is None:
        shuffle = split == "train"
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
