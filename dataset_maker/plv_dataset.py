"""Build scan-separated, forward-aligned BOLD/PLV samples from continuous EEGLAB."""
import argparse
import hashlib
import json
from pathlib import Path
import mne
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, sosfiltfilt, hilbert

BANDS = dict(delta=(.5,4), theta=(4,8), alpha=(8,13), beta=(13,30), gamma=(30,45), broadband=(.5,45))
CHANNELS = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8',
            'P7','P8','FPZ','FZ','CZ','PZ','POZ','OZ','FT9','FT10','TP9','TP10']


def plv_edges(phase):
    u = np.exp(1j*phase)
    matrix = np.abs(u @ u.conj().T / phase.shape[-1])
    return np.clip(matrix[np.triu_indices(len(phase), 1)], 0, 1).astype('float32')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', default='scan_split.yaml')
    p.add_argument('--reference', choices=['average','existing'], required=True)
    p.add_argument('--rois', type=int, default=64)
    p.add_argument('--frames', type=int, default=10)
    p.add_argument('--label-trs', type=int, default=4)
    p.add_argument('--edge-seconds', type=float, default=8.4,
                   help='Discard labels within this distance of recording/filter boundaries; validate by sensitivity analysis.')
    p.add_argument('--only-scans', nargs='*')
    a = p.parse_args()
    root, out = Path(a.data_root), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    split = yaml.safe_load(Path(a.split).read_text())
    owner = {}
    for part, scans in split.items():
        for name in scans:
            subject = name.split('-')[0]
            if subject in owner and owner[subject] != part:
                raise ValueError(f'Subject leakage: {subject}')
            owner[subject] = part
    spec = dict(reference=a.reference, rois=a.rois, frames=a.frames,
                label_trs=a.label_trs, edge_seconds=a.edge_seconds, bands=BANDS,
                channels=CHANNELS, sfreq=200, tr=2.1, filter_order=4,
                filter='continuous sosfiltfilt; reflected Hilbert padding; no fMRI temporal filter',
                alignment='BOLD k:k+frames; EEG [R149[k], R149[k+label_trs])')
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    existing = out/'manifest.json'
    if existing.exists():
        previous = json.loads(existing.read_text())
        if previous['signature'] != signature:
            raise ValueError('Output contains a different configuration; select a new output directory')
    manifest = dict(config=spec, signature=signature, source=str(root.resolve()), scans=[])
    for part, scans in split.items():
        for name in scans:
            if a.only_scans and name not in a.only_scans:
                continue
            print(f'Building {name} ({part})', flush=True)
            raw = mne.io.read_raw_eeglab(root/'EEG_set'/f'{name}_eeg.set', preload=True, verbose='ERROR')
            rename = {c:c.upper().replace("'", '') for c in raw.ch_names}
            raw.rename_channels(rename)
            raw.pick(CHANNELS)
            if raw.ch_names != CHANNELS:
                raw.reorder_channels(CHANNELS)
            events, ids = mne.events_from_annotations(raw, verbose='ERROR')
            times = (events[events[:,2] == ids['R149'],0]-raw.first_samp)/raw.info['sfreq']
            df = pd.read_pickle(root/f'fMRI_difumo{a.rois}'/f'{name}_difumo{a.rois}_roi.pkl')
            bold = df.iloc[:,:-2].to_numpy(dtype='float32')
            if bold.shape != (len(times), a.rois) or not np.isfinite(bold).all():
                raise ValueError(f'{name}: fMRI shape/values mismatch')
            if np.max(np.abs(np.diff(times)-2.1)) > .01:
                raise ValueError(f'{name}: irregular MRI triggers')
            raw.resample(200, verbose='ERROR')
            eeg = raw.get_data(units='uV')
            if not np.isfinite(eeg).all():
                raise ValueError(f'{name}: nonfinite EEG')
            if a.reference == 'average':
                eeg -= eeg.mean(0, keepdims=True)
            fs = 200
            boundaries = [float(t) for t,d in zip(raw.annotations.onset,raw.annotations.description)
                          if 'boundary' in d.lower() and 0<t<eeg.shape[1]/fs]
            if boundaries:
                raise ValueError(f'{name}: internal discontinuities must be segmented before filtering')
            starts = []
            maxk = min(len(times)-a.frames, len(times)-a.label_trs-1)
            for k in range(maxk+1):
                if times[k] >= a.edge_seconds and times[k+a.label_trs] <= eeg.shape[1]/fs-a.edge_seconds:
                    starts.append(k)
            starts = np.asarray(starts, dtype=int)
            if not len(starts):
                raise ValueError(f'{name}: no valid windows')
            labels = np.empty((len(starts),len(BANDS),325),dtype='float32')
            for f, (band,(lo,hi)) in enumerate(BANDS.items()):
                sos = butter(4, [lo,hi], btype='bandpass', fs=fs, output='sos')
                filtered = sosfiltfilt(sos,eeg,axis=-1)
                pad = int(a.edge_seconds*fs)
                padded = np.pad(filtered,((0,0),(pad,pad)),mode='reflect') if pad else filtered
                phase = np.angle(hilbert(padded,axis=-1))
                if pad:
                    phase = phase[:,pad:-pad]
                for n,k in enumerate(starts):
                    i,j = np.rint(times[[k,k+a.label_trs]]*fs).astype(int)
                    labels[n,f] = plv_edges(phase[:,i:j])
            x = np.stack([bold[k:k+a.frames] for k in starts])
            filename = f'{name}.npz'
            np.savez_compressed(out/filename, x=x, y=labels, frame_index=starts,
                                eeg_start=times[starts],eeg_end=times[starts+a.label_trs],
                                signature=signature)
            manifest['scans'].append(dict(scan=name,subject=name.split('-')[0],split=part,
                                          file=filename,windows=len(starts),
                                          peak_abs_uv=float(np.abs(eeg).max())))
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Finished {len(manifest["scans"])} scans: {out}',flush=True)


if __name__ == '__main__':
    main()
