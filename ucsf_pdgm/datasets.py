import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset as hf_load_dataset
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np

_CACHE = {}

# Which labels to keep and how to split — adjust here if the task changes.
_ALLOWED_LABELS = ["Glioblastoma, IDH-wildtype", "Astrocytoma, IDH-mutant", ""]
_N_TRAIN, _N_VAL, _N_TEST, _SEED = 70, 10, 10, 42


def _load_and_aggregate(dataset_id):
    """Loads the raw HF dataset, filters blank slices, aggregates slices into volumes."""
    if dataset_id in _CACHE:
        return _CACHE[dataset_id]

    ds = hf_load_dataset(dataset_id)
    ds = ds.filter(lambda x: np.mean(x["t1c"]) != 0)
    ds = ds["train"].select_columns(["volume_id", "slice_id", "t1c", "tumor_type"])

    volumes = defaultdict(lambda: {"slices": [], "label": None})
    for row in ds:
        vid = row["volume_id"]
        volumes[vid]["slices"].append((row["slice_id"], row["t1c"]))
        volumes[vid]["label"] = row["tumor_type"]

    for vid in volumes:
        volumes[vid]["slices"].sort(key=lambda x: x[0])
        volumes[vid]["slices"] = [img for _, img in volumes[vid]["slices"]]

    _CACHE[dataset_id] = dict(volumes)
    return _CACHE[dataset_id]


def get_dataset(dataset_id, split):

    volumes = _load_and_aggregate(dataset_id)

    volumes = {
        vid: v for vid, v in volumes.items()
        if (v["label"] in _ALLOWED_LABELS) or (not v["label"] and "" in _ALLOWED_LABELS)
    }

    by_label = defaultdict(list)
    for vid, v in volumes.items():
        by_label[v["label"] if v["label"] else ""].append(v)

    train, val, test = [], [], []
    for label, vids in by_label.items():
        trainval, t = train_test_split(vids, test_size=_N_TEST, random_state=_SEED)
        tr, v = train_test_split(trainval, test_size=_N_VAL, random_state=_SEED)
        tr = resample(tr, replace=True, n_samples=_N_TRAIN, random_state=_SEED)
        train.extend(tr)
        val.extend(v)
        test.extend(t)

    return {"train": train, "val": val, "test": test}[split]