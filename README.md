# UnEBOLT

Code for the MIDL 2026 spotlight paper
**"UnEBOLT: A Unified Model for EEG-to-BOLD Translation and Functional Connectivity Reconstruction"**

Yamin Li, Ange Lou, Chang Li, Shiyu Wang, Haatef Pourmotabbed, Ziyuan Xu, Shengchao Zhang,
Dario J. Englot, Soheil Kolouri, Daniel Moyer, Roza G. Bayrak, Catie Chang — Vanderbilt University


<!-- This is a cleaned, single-GPU training release. It predicts 64-ROI DiFuMo fMRI activity from EEG and loads
the public resting-state EEG–fMRI dataset [`NeurdyLab/NeuroBOLT`](https://huggingface.co/datasets/NeurdyLab/NeuroBOLT)
(from the NeuroBOLT paper) directly from HuggingFace with automatic per-scan caching. -->

## Setup

```bash
conda create -n unebolt python=3.9
conda activate unebolt
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

Download the LaBraM checkpoint `labram-base.pth` from the
[LaBraM repo](https://github.com/935963004/LaBraM/tree/main/checkpoints) and place it in `./checkpoints/` —
it initializes the encoder's spatiotemporal module.

## Data

### Option A — use the NeuroBOLT dataset (default)

You do **not** need to download anything manually. On the first training run, the EEG (`.set`) and 64-ROI
DiFuMo fMRI (`.pkl`) files are pulled from HuggingFace into `./data/`:

```
data/
├── EEG_set/sub{NN}-scan{MM}_eeg.set
├── fMRI_difumo64/sub{NN}-scan{MM}_difumo64_roi.pkl
└── cache/sub{NN}-scan{MM}_16s_{nb_roi}roi.pkl   # created automatically after first segmentation
```

With the default `--labels_roi full64`, the 2 trailing confound columns (incl. the cleaned global signal)
are dropped, so the fMRI target is exactly the 64 DiFuMo ROIs.

The dataset is 29 eyes-closed resting-state EEG–fMRI scans from 22 subjects (7 with two scans), 20 min each,
26 EEG channels retained after excluding ECG/EOG/EMG. Each scan is segmented once — a 16 s EEG window
(resampled to 200 Hz) preceding each fMRI TR — and cached under `data/cache/`, so subsequent runs skip
re-segmentation. The train/val/test split is defined in [`scan_split.yaml`](scan_split.yaml). Just run the
[Train](#train) command below.

### Option B — use your own dataset

The model consumes paired **(EEG window, fMRI frame)** samples with this contract:

- **EEG:** float tensor `(N, C, T)` — `N` samples, `C` channels, `T` timepoints. Defaults are `C=26`,
  `T=3200` (a 16 s window at 200 Hz). Channel order must match `ch_names` (see below).
- **fMRI:** float tensor `(N, P)` — one `P`-ROI target per EEG window. Default `P=64` (`--nb_roi`).

To use your own recordings you need to (1) prepare fMRI ROI time series and (2) point the loader at your files:

**1. Prepare fMRI ROI time series.** Reduce your fMRI to a per-ROI time-course table (ROIs × time). If you want
the DiFuMo atlas used here, you can run [`making_difumo.py`](https://github.com/soupeeli/NeuroBOLT) from the
NeuroBOLT repo, which converts an fMRI `.nii` into DiFuMo ROI time series. Any other atlas works too, as long
as the output is an ROI-by-time table.

**2. Rewrite the EEG/fMRI loader.** Adapt `download_vu_dataset()` in
[`dataset_maker/get_datasets.py`](dataset_maker/get_datasets.py) — it loads one scan's EEG `.set`
(`mne.io.read_raw_eeglab`, resampled to 200 Hz, non-EEG channels dropped) and its fMRI ROI table, and returns
`(eeg_raw, fmri_df, roi_labels)`. Repoint the filenames/paths (and, if needed, the channel-exclusion list) to
your own layout. The pairing of each 16 s EEG window with the following fMRI frame is done by
`preproc.epoching_seq2one()` in [`dataset_maker/preproc.py`](dataset_maker/preproc.py); the sync marker used to
time-lock windows to fMRI TRs is `--mri_sync_event` (edit if your EEG uses a different event name).

**3. Update the channel names.** Set `ch_names` in `get_dataset()` in [`main.py`](main.py) to your EEG montage,
in the correct channel order and using 10-20 names (uppercase, e.g. `FP1`, `CZ`, `OZ`) — these drive the
encoder's spatial embeddings. Set `--nb_eegchan` if you have a channel count other than 26. The model scales
EEG by `1/100` internally (see `engine.py`), so provide EEG in µV as the example pipeline does.

> Shortcut: if you already have the paired tensors ready, you can skip the loader entirely and pass a pickle of
> `[train, val, test, args]` (each split a `(eeg_tensor, fmri_tensor)` tuple, shapes `(N,26,3200)` and
> `(N,64)`) via `--prepro_datapath` — this takes the fast path in `prepare_full_dataloader`.

## Train

```bash
python main.py \
  --batch_size 64 --epochs 20 \
  --pretrained_weights ./checkpoints/labram-base.pth \
  --nb_roi 64 --labels_roi full64 --dataset VU \
  --train_test_mode interscan --split_file ./scan_split.yaml \
  --output_dir ./checkpoints/run1/
```

<!-- Trains on a single GPU with AdamW (lr `3e-4`, weight decay `0.05`) and a multi-objective loss:
MSE (`0.8`) + temporal-correlation (`0.1`) + ROI-wise-correlation (`0.1`). The three weights are
configurable via `--loss_mse_weight`, `--loss_temporal_weight`, and `--loss_roi_weight`. -->

<!-- During training, val/test metrics logged to `<output_dir>/log.txt` include temporal correlation
(`corr`), FC-edge correlation / Pixcorr (`conn_spatial_corr`), connectivity MSE (`conn_mse`), and
edge-detection F1 at the top-25% / top-50% thresholds (`f1_025`, `f1_05`). -->

## Inference

To evaluate a trained checkpoint on a scan split (default: `test`) and report the full metric set
per scan and aggregated as `mean(std)`:

```bash
python inference.py \
  --ckpt ./checkpoints/run1/checkpoint-<...>.pth \
  --nb_roi 64 --dataset VU \
  --split_file ./scan_split.yaml --output_dir ./checkpoints/run1/
```

It reports **Tcorr** (temporal correlation), **Pixcorr** (FC-edge correlation), **ConnMSE**
(FC-edge MSE), and **F1@0.25 / F1@0.5** (edge-detection F1 on the strongest 25% / 50% of FC edges),
and writes:
- `inference_metrics.csv` — per-scan rows plus a trailing `mean(std)` row,
- `inference_predictions.npz` — concatenated predictions (`y_hat`) and ground truth (`y_test`),
- `inference_figures/<scan>_roi<nb_roi>.png` — predicted-vs-ground-truth FC heatmaps (pass
  `--no_figures` to skip).

Use `--eval_split val` to evaluate the validation split, and `--device cpu` to run on CPU.


## Reference

```bibtex
@inproceedings{li2026unebolt,
  title={UnEBOLT: A Unified Model for EEG-to-BOLD Translation and Functional Connectivity Reconstruction},
  author={Li, Yamin and Lou, Ange and Li, Chang and Wang, Shiyu and Pourmotabbed, Haatef and Xu, Ziyuan and Zhang, Shengchao and Englot, Dario J and Kolouri, Soheil and Moyer, Daniel and Bayrak, Roza G and Chang, Catie},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  year={2026}
}
```

The dataset and multi-dimensional encoder build on NeuroBOLT (Li et al., NeurIPS 2024). Built upon the LaBraM,
BIOT, BEiT-v2, timm, DeiT, and DINO codebases — we thank the authors.

