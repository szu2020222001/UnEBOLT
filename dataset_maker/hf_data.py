"""
HuggingFace data pipeline for NeuroBOLT (UnEBOLT_clean).

The public dataset ``NeurdyLab/NeuroBOLT`` ships raw EEGLAB ``.set`` files plus
64-ROI DiFuMo fMRI pickles:

    <data_root>/
    ├── EEG_set/<patient>_eeg.set
    └── fMRI_difumo{nb_roi}/<patient>_difumo{nb_roi}_roi.pkl

where ``patient = sub{sub:02d}-scan{scan:02d}`` (29 scans, 22 subjects).

This module:
  1. downloads the dataset to ``data_root`` if it is not already there,
  2. segments each scan into (16 s EEG window -> single fMRI TR) samples via
     ``preproc.epoching_seq2one``, caching the result per scan under
     ``<data_root>/cache/`` so a scan is only segmented once,
  3. assembles the per-scan caches into train / val / test ``TensorDataset``s
     according to a split spreadsheet.

The output of :func:`build_split_datasets` matches
``get_datasets.prepare_full_dataloader``'s ``--prepro_datapath`` fast path, so
the two are interchangeable.
"""

import os
import re
import pickle

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import butter, filtfilt

from dataset_maker import preproc
from dataset_maker.get_datasets import download_vu_dataset, convert_to_tensor

HF_REPO_ID = "NeurdyLab/NeuroBOLT"

# Segmentation config (16 s EEG window preceding each fMRI TR, 200 Hz -> 3200 samples)
TMIN = -16
TMAX = 0
CROP = 3200


def ensure_dataset(data_root, repo_id=HF_REPO_ID, difumo_dim=64):
    """Download the HuggingFace dataset into ``data_root`` if not already present."""
    eeg_dir = os.path.join(data_root, "EEG_set")
    fmri_dir = os.path.join(data_root, f"fMRI_difumo{difumo_dim}")
    if os.path.isdir(eeg_dir) and os.path.isdir(fmri_dir):
        return
    from huggingface_hub import snapshot_download
    os.makedirs(data_root, exist_ok=True)
    print(f"[hf_data] '{eeg_dir}' not found; downloading {repo_id} -> {data_root} ...")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=data_root)
    print("[hf_data] download complete.")


def _process_fmri_series(df_fmri, labels_roi_full, args):
    """ROI selection + <0.15 Hz low-pass + z-score.

    Returns the normalized ``(#roi, #time)`` array, before epoching/frame selection.
    """
    fmri_np = df_fmri.to_numpy().T
    df = pd.DataFrame(data=fmri_np.T, columns=labels_roi_full)
    if args.labels_roi in ['full64', 'full128', 'full256', 'full512', 'full1024']:
        df_filter = df.iloc[:, :-2]  # keep the DiFuMo ROIs, drop the trailing confound columns
    else:
        df_filter = df[args.labels_roi]
    fmri_np = df_filter.to_numpy().T
    if fmri_np.ndim < 2:
        fmri_np = fmri_np[np.newaxis, ...]

    fs = 1 / args.TR
    nyquist = 0.5 * fs
    low = 0.15 / nyquist
    b, a = butter(N=5, Wn=low, btype='low', analog=False)
    fmri_np = filtfilt(b, a, fmri_np, axis=1)
    fmri_norm, _ = preproc.normalize_data(fmri_np, mode='zscore')
    return fmri_norm


def _read_fmri(sub, scan, args):
    """Read only the DiFuMo fMRI ROI pickle for the requested resolution (no EEG .set)."""
    patient = 'sub{:02d}-scan{:02d}'.format(int(sub), int(scan))
    path = os.path.join(args.dataset_root, f'fMRI_difumo{args.difumo_dim}',
                        f'{patient}_difumo{args.difumo_dim}_roi.pkl')
    df_fmri = pd.read_pickle(path)
    return df_fmri, df_fmri.columns.to_list()


def _segment_scan(sub, scan, args):
    """Full segmentation for one scan (reads the EEG .set + fMRI pickle).

    Returns ``(eeg_list, fmri_list, select_ind)``. ``select_ind`` (the surviving
    epoch indices, from ``eeg_epoch.selection``) lets the fMRI target for *other*
    ROI resolutions be rebuilt later without touching the EEG.
    """
    eeg_raw, fmri, labels_roi_full = download_vu_dataset(
        sub, scan, args.dataset_root, output_eegmneraw=True, difumo_dim=args.difumo_dim)
    eeg_raw.load_data()
    eeg_raw.filter(l_freq=0.5, h_freq=None)
    fmri_norm = _process_fmri_series(fmri, labels_roi_full, args)
    data_epoch, eeg_epoch = preproc.epoching_seq2one(
        eeg_raw, fmri_norm, TMIN, TMAX, args.mri_sync_event, ifnorm=0, crop=CROP)
    return data_epoch["eeg"], data_epoch["fmri"], np.asarray(eeg_epoch.selection)


def _segment_fmri(sub, scan, args, select_ind):
    """Build only the fMRI target for one scan/resolution, aligned to ``select_ind`` (no EEG .set)."""
    df_fmri, labels_roi_full = _read_fmri(sub, scan, args)
    fmri_norm = _process_fmri_series(df_fmri, labels_roi_full, args)
    fmri_epoch = fmri_norm[:, select_ind]
    return [np.array(sample) for sample in fmri_epoch.T]


def _sanitize(name):
    return re.sub(r'[^A-Za-z0-9]+', '', str(name))


def load_or_build_scan(sub, scan, args, cache_dir):
    """Return (eeg_list, fmri_list) for one scan.

    The EEG windows are cached once per scan (``<patient>_16s_eeg.pkl``) and shared
    across ROI resolutions; the fMRI target is cached per resolution
    (``<patient>_16s_difumo{dim}_{labels}_fmri.pkl``). An older combined cache
    (``<patient>_16s_{nb_roi}roi.pkl``) is migrated in place when present, so no
    re-segmentation is needed for data cached by earlier versions.
    """
    patient = 'sub{:02d}-scan{:02d}'.format(int(sub), int(scan))
    eeg_cache = os.path.join(cache_dir, f"{patient}_16s_eeg.pkl")
    fmri_cache = os.path.join(
        cache_dir, f"{patient}_16s_difumo{args.difumo_dim}_{_sanitize(args.labels_roi)}_fmri.pkl")
    legacy_cache = os.path.join(cache_dir, f"{patient}_16s_{args.nb_roi}roi.pkl")
    os.makedirs(cache_dir, exist_ok=True)

    def _load(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save(path, obj):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    eeg_blob = _load(eeg_cache)     # {"eeg": eeg_list, "select_ind": ...} or None
    fmri_list = _load(fmri_cache)   # fmri_list or None
    if eeg_blob is not None and fmri_list is not None:
        return eeg_blob["eeg"], fmri_list  # fast path: both decoupled caches present

    legacy = _load(legacy_cache)    # (eeg_list, fmri_list) or None
    legacy_matches = legacy is not None and len(np.atleast_1d(legacy[1][0])) == args.nb_roi

    eeg_list = eeg_blob["eeg"] if eeg_blob is not None else None
    select_ind = eeg_blob.get("select_ind") if eeg_blob is not None else None

    # Migrate the (resolution-independent) EEG from the old combined cache, no re-segmentation.
    if eeg_list is None and legacy is not None:
        eeg_list, select_ind = legacy[0], None
        _save(eeg_cache, {"eeg": eeg_list, "select_ind": None})

    # Migrate the fMRI target if the legacy cache is the same ROI resolution.
    if fmri_list is None and legacy_matches:
        fmri_list = legacy[1]

    # If EEG is still missing, or we need select_ind to build the fMRI, run one full pass.
    if eeg_list is None or (fmri_list is None and select_ind is None):
        seg_eeg, seg_fmri, select_ind = _segment_scan(sub, scan, args)
        if eeg_list is None:
            eeg_list = seg_eeg
        if fmri_list is None:
            fmri_list = seg_fmri
        _save(eeg_cache, {"eeg": eeg_list, "select_ind": select_ind})

    # New resolution with the EEG already cached: build only the fMRI target (no EEG .set read).
    if fmri_list is None:
        fmri_list = _segment_fmri(sub, scan, args, select_ind)

    _save(fmri_cache, fmri_list)
    print(f"[hf_data] {patient}: {len(eeg_list)} samples "
          f"(eeg cache + difumo{args.difumo_dim}/{_sanitize(args.labels_roi)} fmri cache)")
    return eeg_list, fmri_list


def build_split_datasets(args):
    """Build (train, test, val) TensorDatasets from the HuggingFace dataset.

    Reads the split from the YAML file ``args.split_file`` -- a mapping with
    ``train`` / ``val`` / ``test`` keys, each a list of scan names, e.g.::

        train:
          - sub01-scan01
        val:
          - sub06-scan01
        test:
          - sub07-scan01

    Each scan is segmented (or loaded from cache) and concatenated per split.
    """
    ensure_dataset(args.dataset_root, difumo_dim=args.difumo_dim)
    cache_dir = os.path.join(args.dataset_root, "cache")

    with open(args.split_file, "r", encoding="utf-8") as f:
        split = yaml.safe_load(f)

    eeg_data = {"train": [], "val": [], "test": []}
    fmri_data = {"train": [], "val": [], "test": []}

    for split_name in ("train", "val", "test"):
        for scan in split.get(split_name) or []:
            sub_ind, scan_ind = re.findall(r'\d+', str(scan))
            print(f"[hf_data] preparing {scan} ({split_name}) ...")
            eeg_list, fmri_list = load_or_build_scan(sub_ind, scan_ind, args, cache_dir)
            eeg_data[split_name] += eeg_list
            fmri_data[split_name] += fmri_list

    def make_dataset(split):
        eeg_tensor = convert_to_tensor(eeg_data[split])
        fmri_tensor = convert_to_tensor(fmri_data[split]).squeeze()
        return torch.utils.data.TensorDataset(eeg_tensor, fmri_tensor)

    train_dataset = make_dataset("train")
    val_dataset = make_dataset("val")
    test_dataset = make_dataset("test")

    print("Datasets successfully created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    return train_dataset, test_dataset, val_dataset
