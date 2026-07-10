import mne
import os
import json

import numpy as np
import pandas as pd
import re

from dataset_maker import preproc
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Sampler
import random
import math
import pickle


def download_vu_dataset(sub, scan, path_to_dataset, output_eegmneraw=False, out_eegchan='EEG', difumo_dim=64):
    # File layout follows the public HuggingFace dataset NeurdyLab/NeuroBOLT
    # (difumo_dim selects the DiFuMo resolution, e.g. 64 -> fMRI_difumo64):
    #   <root>/EEG_set/sub{NN}-scan{MM}_eeg.set
    #   <root>/fMRI_difumo{difumo_dim}/sub{NN}-scan{MM}_difumo{difumo_dim}_roi.pkl
    patient = 'sub{:02d}-scan{:02d}'.format(int(sub), int(scan))
    eeg_path_set_file = os.path.join(path_to_dataset, 'EEG_set', f'{patient}_eeg.set')
    fMRI_path_difumo = os.path.join(path_to_dataset, f'fMRI_difumo{difumo_dim}',
                                    f'{patient}_difumo{difumo_dim}_roi.pkl')

    new_fps = 200

    # ----- Load and preprocess EEG -----
    raw = mne.io.read_raw_eeglab(eeg_path_set_file)
    original_fps = raw.info['sfreq']
    if original_fps != new_fps:
        raw.resample(new_fps)
    if len(raw.ch_names) > 32:
        vector_exclude = ['EOG1', 'EOG2', 'EMG1', 'EMG2', 'EMG3', 'ECG',
                          'CWL1', 'CWL2', 'CWL3', 'CWL4']
    else:
        vector_exclude = ['EOG1', 'EOG2', 'EMG1', 'EMG2', 'EMG3', 'ECG']
    # only drop channels that are actually present (public .set files may already be EEG-only)
    vector_exclude = [ch for ch in vector_exclude if ch in raw.ch_names]

    if output_eegmneraw == True:
        if out_eegchan == 'EEG':
            raw.drop_channels(vector_exclude)
        elif out_eegchan == 'EMG':
            raw.pick_channels(['EMG1', 'EMG2', 'EMG3'])
        df_eeg = raw
    else:
        df_eeg = raw.to_data_frame()
        df_eeg = df_eeg.drop(vector_exclude, axis=1)

    # ----- Load fMRI ROI time series (Difumo atlas) -----
    df_fmri = pd.read_pickle(fMRI_path_difumo)

    ROI_labels = df_fmri.columns.to_list()

    return df_eeg, df_fmri, ROI_labels


def convert_to_tensor(data):
    tensors = [torch.tensor(arr) for arr in data]
    return torch.stack(tensors, dim=0)


def prepare_onesub_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # make sure the HuggingFace dataset is available locally
    from dataset_maker import hf_data
    hf_data.ensure_dataset(args.dataset_root, difumo_dim=args.difumo_dim)

    root = args.dataset_root
    tr = args.TR
    event_sync_name = args.mri_sync_event

    tmin = -16
    tmax = 0
    crop = 3200  # 16 * 200

    sub_ind, scan_ind = re.findall(r'\d+', args.dataname)
    sub_ind, scan_ind = int(sub_ind), int(scan_ind)

    eeg_raw, fmri, labels_roi_full = download_vu_dataset(sub_ind, scan_ind, root, output_eegmneraw=True,
                                                         difumo_dim=args.difumo_dim)
    fmri_np = fmri.to_numpy().T
    # extract ROI

    df = pd.DataFrame(data=fmri_np.T, columns=labels_roi_full)
    if args.labels_roi in ['full64', 'full128', 'full256']:
        df_filter = df.iloc[:, :-2]  # keep all DiFuMo ROIs, drop the 2 trailing confound columns
    else:
        df_filter = df[args.labels_roi]

    fmri_np = df_filter.to_numpy().T
    if len(fmri_np.shape) < 2:
        fmri_np = fmri_np[np.newaxis, ...]

    # Optional: add filtering, filter below 0.15Hz
    # This part can also be moved to making_difumo.py
    fs = 1 / tr
    nyquist = 0.5 * fs
    low = 0.15 / nyquist  # Low cutoff frequency (as a fraction of Nyquist frequency)
    b, a = butter(N=5, Wn=low, btype='low', analog=False)  # Using a 5th order filter
    # Apply the filter
    fmri_np = filtfilt(b, a, fmri_np, axis=1)

    fmri_norm, _ = preproc.normalize_data(fmri_np)

    # epoching data, and divide them into train, val and test
    data_epoch, eeg_info = preproc.epoching_seq2one(eeg_raw, fmri_norm,
                                                    tmin, tmax, event_sync_name, ifnorm=0,
                                                    crop=crop)
    traincrop = int(0.8 * len(data_epoch["eeg"]))
    valcrop = int(0.1 * len(data_epoch["eeg"])) + traincrop

    eeg_train = data_epoch["eeg"][:traincrop]
    fmri_train = data_epoch["fmri"][:traincrop]

    # consider the auto-correlaiton of fMRI, preventing data leakage
    t_overlap = 20 if abs(tmin) <= 20 else abs(tmin)  # Length of HRF consideration
    N_overlap = math.ceil(t_overlap / tr)

    traincrop += N_overlap
    eeg_val = data_epoch["eeg"][traincrop:valcrop]
    fmri_val = data_epoch["fmri"][traincrop:valcrop]

    valcrop += N_overlap
    eeg_test = data_epoch["eeg"][valcrop:]
    fmri_test = data_epoch["fmri"][valcrop:]

    eeg_train_tensor = convert_to_tensor(eeg_train)
    eeg_val_tensor = convert_to_tensor(eeg_val)
    eeg_test_tensor = convert_to_tensor(eeg_test)

    fmri_train_tensor = convert_to_tensor(fmri_train).squeeze()
    fmri_val_tensor = convert_to_tensor(fmri_val).squeeze()
    fmri_test_tensor = convert_to_tensor(fmri_test).squeeze()

    train_dataset = torch.utils.data.TensorDataset(eeg_train_tensor, fmri_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(eeg_val_tensor, fmri_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(eeg_test_tensor, fmri_test_tensor)

    return train_dataset, test_dataset, val_dataset


def prepare_full_dataloader(args):
    if args.prepro_datapath:
        # Load preprocessed data
        try:
            with open(args.prepro_datapath, 'rb') as file:
                train_data, val_data, test_data, args_data = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {args.prepro_datapath}")
        except Exception as e:
            raise RuntimeError(f"Error loading data from {args.prepro_datapath}: {e}")

        # Unpack data tensors
        eeg_train_tensor, fmri_train_tensor = train_data
        eeg_val_tensor, fmri_val_tensor = val_data
        eeg_test_tensor, fmri_test_tensor = test_data

        # Create TensorDatasets
        train_dataset = torch.utils.data.TensorDataset(eeg_train_tensor, fmri_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(eeg_val_tensor, fmri_val_tensor)
        test_dataset = torch.utils.data.TensorDataset(eeg_test_tensor, fmri_test_tensor)

        # Optional: Log dataset creation
        print("Datasets successfully created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        return train_dataset, test_dataset, val_dataset

    # No preprocessed pickle given: build from the public HuggingFace dataset
    # (downloads on first use, then caches each scan under <dataset_root>/cache/).
    from dataset_maker import hf_data
    return hf_data.build_split_datasets(args)


class SequentialBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.n = len(data_source)

        # total number of batches
        if self.drop_last:
            self.num_batches = self.n // self.batch_size
        else:
            self.num_batches = (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # start index of every batch
        starts = [i * self.batch_size for i in range(self.num_batches)]

        # shuffle the batch order
        if self.shuffle:
            random.shuffle(starts)

        # build the contiguous indices of each batch
        batches = []
        for start in starts:
            end = start + self.batch_size
            # clamp the last batch if it runs past the end of the data
            if end > self.n:
                end = self.n
            batch_indices = list(range(start, end))
            batches.append(batch_indices)

        return iter(batches)

    def __len__(self):
        return self.num_batches


class CyclicSequentialBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False, shuffle=True, seed=1124):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.n = len(data_source)
        self.seed = seed  # note: currently unused (batch order is not seeded here)

        # theoretical max number of batches (with cyclic wrap-around padding)
        self.num_batches = (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # random starting offset in [0, batch_size - 1]
        start_offset = random.randint(0, self.batch_size - 1) if self.shuffle else 0

        # generate all sample indices, wrapping around cyclically
        indices = []
        current = start_offset
        for _ in range(self.num_batches * self.batch_size):
            indices.append(current % self.n)
            current += 1

        # slice the indices into batches
        batches = []
        for i in range(self.num_batches):
            batch = indices[i * self.batch_size: (i + 1) * self.batch_size]
            # keep the batch if incomplete batches are allowed or it is full
            if not self.drop_last or len(batch) == self.batch_size:
                batches.append(batch)

        # shuffle the batch order (optional)
        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        return self.num_batches if self.drop_last else (self.n + self.batch_size - 1) // self.batch_size