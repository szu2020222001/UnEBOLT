import io
import os
import math
import time
import json
import glob
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict

from pathlib import Path

import torch
from torch import inf

from tensorboardX import SummaryWriter

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

standard_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """No-op on single GPU (kept for API compatibility)."""
        return

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def update_image(self, head='images', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.writer.add_image(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


# ---- Single-GPU stubs (kept so the rest of the code that references these
#      helpers continues to work without a distributed backend) ----
def is_dist_avail_and_initialized():
    return False


def get_world_size():
    return 1


def get_rank():
    return 0


def is_main_process():
    return True


def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,
                 layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0, layer_names=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device

    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))

        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")

    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None,
               save_ckpt_freq=1):
    epoch_name = str(epoch)
    if args.labels_roi == "Heschl’s gyrus":
        label_roi = "Heschl gyrus_smooth_16s"
    else:
        label_roi = args.labels_roi + "_smooth_16s"

    if args.train_test_mode == 'intrascan':
        subdir_name = f"{args.dataset}-{args.train_test_mode}-{args.dataname}-{label_roi}-{args.model}-{args.lr}-{args.batch_size}"
    else:
        subdir_name = f"{args.dataset}-{args.train_test_mode}-full-{label_roi}-{args.model}-{args.lr}-{args.batch_size}"

    output_dir = Path(args.output_dir + subdir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = [output_dir / 'checkpoint.pth']
    if epoch is not None:
        checkpoint_paths = [output_dir / ('checkpoint-epoch%s.pth' % epoch_name), ]
    elif (epoch + 1) % save_ckpt_freq == 0:
        checkpoint_paths.append(output_dir / ('checkpoint-epoch%s.pth' % epoch_name))

    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
        }
        if loss_scaler is not None:
            to_save['scaler'] = loss_scaler.state_dict()

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)

        save_on_master(to_save, checkpoint_path)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)

    if args.auto_resume and len(args.resume) == 0:
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint.pth'))
        if len(all_checkpoints) > 0:
            args.resume = os.path.join(output_dir, 'checkpoint.pth')
        else:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])  # strict: bool=True, , strict=False
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
            args.start_epoch = 1  # checkpoint['epoch'] + 1
            if hasattr(args, 'model_ema') and args.model_ema:
                _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def get_input_chans(ch_names):
    input_chans = [0]  # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans



def get_metrics(output, target, metrics, is_binary):
    if is_binary:
        if type(output) == list:
            mse_all = [mse_metric(x, y) for x, y in zip(output, target)]
            corr_all = [corr_metric(x, y) for x, y in zip(output, target)]
            results = {
                "mse": np.mean(mse_all),
                "corr": np.mean(corr_all),
            }
        else:
            results = {
                "mse": mse_metric(output, target),
                "corr": corr_metric(output, target)
            }
    else:
        if type(output) == list:
            mse_all = []
            corr_all = []
            concorr_all = []
            connmse_all = []
            f1_025_all = []
            f1_05_all = []
            for x, y in zip(output, target):
                num_rois = x.shape[1]
                mse_scores = [mse_metric(x[:, roi], y[:, roi]) for roi in range(num_rois)]
                corr_scores = [corr_metric(x[:, roi], y[:, roi]) for roi in range(num_rois)]

                fc_pred, fc_true = fc_matrix(x), fc_matrix(y)
                mse_all.append(np.mean(mse_scores))
                corr_all.append(np.mean(corr_scores))
                concorr_all.append(pixcorr(fc_pred, fc_true))
                connmse_all.append(conn_mse(fc_pred, fc_true))
                f1_025_all.append(f1_edges(fc_pred, fc_true, 0.25))
                f1_05_all.append(f1_edges(fc_pred, fc_true, 0.5))

            results = {
                "mse": np.mean(mse_all),
                "corr": np.mean(corr_all),
                "conn_spatial_corr": np.mean(concorr_all),
                "conn_mse": np.mean(connmse_all),
                "f1_025": np.mean(f1_025_all),
                "f1_05": np.mean(f1_05_all),
            }
        else:
            # output shape: batch x # roi
            num_rois = output.shape[1]
            mse_scores = [mse_metric(output[:, roi], target[:, roi]) for roi in range(num_rois)]
            corr_scores = [corr_metric(output[:, roi], target[:, roi]) for roi in range(num_rois)]

            fc_pred, fc_true = fc_matrix(output), fc_matrix(target)
            results = {
                "mse": np.mean(mse_scores),
                "corr": np.mean(corr_scores),
                "conn_spatial_corr": pixcorr(fc_pred, fc_true),
                "conn_mse": conn_mse(fc_pred, fc_true),
                "f1_025": f1_edges(fc_pred, fc_true, 0.25),
                "f1_05": f1_edges(fc_pred, fc_true, 0.5),
            }

    return results


def corr_metric(x, y):
    """
    x and y - 1D vectors
    """
    assert x.shape == y.shape, f'{x.shape} and {y.shape}'
    r = np.corrcoef(x.squeeze(), y.squeeze())[0, 1]
    return r


def mse_metric(x, y):
    assert x.shape == y.shape, f'{x.shape} and {y.shape}'
    return mean_squared_error(x, y)


def spatial_corr(x, y):
    """
    x and y - 2D vectors, batchxroi
    """
    assert x.shape == y.shape, f'{x.shape} and {y.shape}'
    nbroi = x.shape[1]

    conn_map = np.corrcoef(x.T, y.T)
    triu_indices = np.tril_indices(x.shape[1], k=-1)  # Lower triangle indices
    spatial_corr = np.corrcoef(conn_map[:nbroi, :nbroi][triu_indices], conn_map[-nbroi:, -nbroi:][triu_indices])[0, 1]
    return spatial_corr


# ---- Functional-connectivity (FC) metrics ----
# An FC matrix is the ROI-by-ROI Pearson correlation of a scan's ROI time series.
# All FC metrics operate on the strict upper triangle (off-diagonal edges).

def fc_matrix(ts):
    """ts: (T, R) ROI time series -> (R, R) functional-connectivity matrix."""
    return np.corrcoef(ts.T)


def fc_edges(fc):
    """Upper-triangle (excluding diagonal) edges of an FC matrix."""
    return fc[np.triu_indices(fc.shape[0], k=1)]


def pixcorr(fc_pred, fc_true):
    """Pixel-wise correlation between predicted and true FC edges (paper's Pixcorr).

    Equivalent to ``spatial_corr`` above, but takes the FC matrices directly.
    """
    return np.corrcoef(fc_edges(fc_pred), fc_edges(fc_true))[0, 1]


def conn_mse(fc_pred, fc_true):
    """MSE between predicted and true FC edges (paper's ConnMSE)."""
    return mean_squared_error(fc_edges(fc_pred), fc_edges(fc_true))


def f1_edges(fc_pred, fc_true, top_frac):
    """Edge-detection F1 at a top-fraction threshold (paper's F1-0.25 / F1-0.5).

    Each edge vector is binarized by keeping its own strongest ``top_frac`` edges
    (per-matrix percentile), then F1 is computed between the two binary masks.
    """
    ep, et = fc_edges(fc_pred), fc_edges(fc_true)
    bt = (et >= np.percentile(et, 100 * (1 - top_frac))).astype(int)
    bp = (ep >= np.percentile(ep, 100 * (1 - top_frac))).astype(int)
    tp = int(((bt == 1) & (bp == 1)).sum())
    fp = int(((bt == 0) & (bp == 1)).sum())
    fn = int(((bt == 1) & (bp == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0