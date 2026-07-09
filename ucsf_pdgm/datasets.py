import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset as hf_load_dataset
from collections import defaultdict
import numpy as np


def get_dataset(dataset_id, split="train"):
    """
    Loads a Hugging Face dataset, filters out blank slices, and aggregates
    slices into volumes.

    Returns: dict of {volume_id: {"slices": [PIL.Image, ...], "label": str or None}}
    """
    ds = hf_load_dataset(dataset_id)
    ds = ds.filter(lambda x: np.mean(x["t1c"]) != 0)
    ds = ds[split].select_columns(["volume_id", "slice_id", "t1c", "tumor_type"])

    volumes = defaultdict(lambda: {"slices": [], "label": None})
    for row in ds:
        vid = row["volume_id"]
        volumes[vid]["slices"].append((row["slice_id"], row["t1c"]))
        volumes[vid]["label"] = row["tumor_type"]

    for vid in volumes:
        volumes[vid]["slices"].sort(key=lambda x: x[0])
        volumes[vid]["slices"] = [img for _, img in volumes[vid]["slices"]]

    return dict(volumes)