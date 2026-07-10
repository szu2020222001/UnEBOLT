"""
UnEBOLT inference.

Loads a trained UnEBOLT checkpoint and evaluates it on a scan split (default: test),
reporting the full set of metrics from the paper for each scan and aggregated as
mean(std):

  - Tcorr    : temporal correlation (per-ROI Pearson, averaged over ROIs)
  - Pixcorr  : correlation between predicted and true functional-connectivity (FC) edges
  - ConnMSE  : MSE between predicted and true FC edges
  - F1@0.25  : edge-detection F1 on the top-25% strongest FC edges
  - F1@0.5   : edge-detection F1 on the top-50% strongest FC edges

Outputs (under --output_dir):
  - inference_metrics.csv        : per-scan + mean/std rows
  - inference_predictions.npz    : concatenated y_hat / y_test
  - inference_figures/<scan>.png : predicted-vs-ground-truth FC heatmaps (unless --no_figures)

Example:
  python inference.py --resume ./checkpoints/run1/checkpoint-<...>.pth \
    --nb_roi 64 --difumo_dim 64 --split_file ./scan_split.yaml \
    --output_dir ./checkpoints/run1/
"""

import os
import re

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")  # headless-safe; no interactive backend
import matplotlib.pyplot as plt
from einops import rearrange

import utils
import models.unebolt_model  # noqa: F401  (registers the 'unebolt' model)
from main import get_args, get_models, VU_CH_NAMES
from dataset_maker import hf_data
from dataset_maker.get_datasets import convert_to_tensor

METRIC_COLS = ["Tcorr", "Pixcorr", "ConnMSE", "F1@0.25", "F1@0.5"]


def load_model(args, device):
    """Build the UnEBOLT model and load a trained checkpoint's weights."""
    model = get_models(args)
    checkpoint = torch.load(args.resume, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_scan(model, eeg, input_chans, device, batch_size):
    """Run one scan's EEG (N, C, T) through the model in chunks -> (N, R) predictions."""
    preds = []
    for i in range(0, len(eeg), batch_size):
        x = eeg[i:i + batch_size].float().to(device, non_blocking=True) / 100
        x = rearrange(x, "B N (A T) -> B N A T", T=200)
        out = model(x, input_chans=input_chans, input_chans_spect=None)
        preds.append(out.cpu())
    return torch.cat(preds, dim=0).numpy()


def scan_metrics(pred, true):
    """All five metrics for one scan. pred/true: (N_timepoints, R)."""
    num_rois = pred.shape[1]
    tcorr = float(np.mean([utils.corr_metric(pred[:, r], true[:, r]) for r in range(num_rois)]))
    fc_pred, fc_true = utils.fc_matrix(pred), utils.fc_matrix(true)
    return {
        "Tcorr": tcorr,
        "Pixcorr": float(utils.pixcorr(fc_pred, fc_true)),
        "ConnMSE": float(utils.conn_mse(fc_pred, fc_true)),
        "F1@0.25": float(utils.f1_edges(fc_pred, fc_true, 0.25)),
        "F1@0.5": float(utils.f1_edges(fc_pred, fc_true, 0.5)),
    }, fc_pred, fc_true


def save_fc_figure(fc_pred, fc_true, scan, metrics, fig_dir, nb_roi):
    """2x2 FC heatmaps: raw pred/GT (top) and top-25% thresholded pred/GT (bottom)."""
    thr_pred = np.percentile(utils.fc_edges(fc_pred), 75)
    thr_true = np.percentile(utils.fc_edges(fc_true), 75)
    panels = [
        (fc_pred, "Predicted FC"),
        (fc_true, "Ground-truth FC"),
        (fc_pred * (fc_pred >= thr_pred), "Predicted FC (top 25%)"),
        (fc_true * (fc_true >= thr_true), "Ground-truth FC (top 25%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, (mat, title) in zip(axes.ravel(), panels):
        im = ax.imshow(mat, vmin=0.2, vmax=1.0, cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.suptitle(f"{scan}  |  Tcorr={metrics['Tcorr']:.3f}  Pixcorr={metrics['Pixcorr']:.3f}  "
                 f"ConnMSE={metrics['ConnMSE']:.3f}", fontsize=11)
    out = os.path.join(fig_dir, f"{scan}_roi{nb_roi}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_inference(args):
    device = torch.device(args.device)
    model = load_model(args, device)
    input_chans = utils.get_input_chans(VU_CH_NAMES)

    with open(args.split_file, "r", encoding="utf-8") as f:
        split = yaml.safe_load(f)
    scans = split.get(args.eval_split) or []
    if not scans:
        raise ValueError(f"No scans found for split '{args.eval_split}' in {args.split_file}")

    hf_data.ensure_dataset(args.dataset_root, difumo_dim=args.difumo_dim)
    cache_dir = os.path.join(args.dataset_root, "cache")
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "inference_figures")
    if not args.no_figures:
        os.makedirs(fig_dir, exist_ok=True)

    rows, all_pred, all_true = [], [], []
    for scan in scans:
        sub_ind, scan_ind = re.findall(r"\d+", str(scan))
        eeg_list, fmri_list = hf_data.load_or_build_scan(sub_ind, scan_ind, args, cache_dir)
        eeg = convert_to_tensor(eeg_list)                       # (N, C, T)
        true = convert_to_tensor(fmri_list).squeeze().numpy()   # (N, R)
        pred = predict_scan(model, eeg, input_chans, device, args.batch_size)

        metrics, fc_pred, fc_true = scan_metrics(pred, true)
        rows.append({"scan": scan, **metrics})
        all_pred.append(pred); all_true.append(true)
        if not args.no_figures:
            save_fc_figure(fc_pred, fc_true, scan, metrics, fig_dir, args.nb_roi)
        print(f"[{scan}] " + "  ".join(f"{k}={metrics[k]:.4f}" for k in METRIC_COLS))

    df = pd.DataFrame(rows)
    summary = {"scan": "Mean(std)",
               **{c: f"{df[c].mean():.4f}({df[c].std():.4f})" for c in METRIC_COLS}}

    print("\n===== Per-scan metrics =====")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n===== Aggregate (mean(std)) =====")
    print("  ".join(f"{c}: {summary[c]}" for c in METRIC_COLS))

    # Persist: per-scan rows + a trailing mean(std) row
    csv_path = os.path.join(args.output_dir, "inference_metrics.csv")
    pd.concat([df, pd.DataFrame([summary])], ignore_index=True).to_csv(csv_path, index=False)
    np.savez(os.path.join(args.output_dir, "inference_predictions.npz"),
             y_hat=np.concatenate(all_pred, axis=0), y_test=np.concatenate(all_true, axis=0))
    print(f"\nSaved metrics -> {csv_path}")
    if not args.no_figures:
        print(f"Saved FC figures -> {fig_dir}")


if __name__ == "__main__":
    args = get_args()
    if not args.resume:
        raise SystemExit("inference.py requires --resume <checkpoint.pth>")
    run_inference(args)
