"""
UnEBOLT: A Unified Model for EEG-to-BOLD Translation and Functional Connectivity Reconstruction
By Yamin Li

UnEBOLT is built upon the LaBraM, BIOT, NeuroBOLT, BEiT-v2, timm, DeiT, and DINO codebases,
we extend our gratitude to the authors for their contributions:

- https://github.com/935963004/LaBraM
- https://github.com/ycq091044/BIOT
- https://github.com/soupeeli/NeuroBOLT
- https://github.com/microsoft/unilm/tree/master/beitv2
- https://github.com/rwightman/pytorch-image-models/tree/master/timm
- https://github.com/facebookresearch/deit/
- https://github.com/facebookresearch/dino
"""

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import yaml

from pathlib import Path
from collections import OrderedDict
from timm.models import create_model
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from engine import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from customized_losses import CustomMSECorrLoss
from dataset_maker import get_datasets
from dataset_maker.get_datasets import CyclicSequentialBatchSampler
import models.unebolt_model


def get_args():
    parser = argparse.ArgumentParser('UnEBOLT intra- and inter-subject training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int,
                        help='save a checkpoint every N epochs (1 = every epoch)')

    # Model parameters
    parser.add_argument('--model', default='unebolt', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=False)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.1, metavar='PCT',
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--attn_drop_rate', type=float, default=0., metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.65)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (default: 1e-6)')

    # Loss weights (multi-objective loss: MSE + temporal-correlation + ROI-wise-correlation)
    parser.add_argument('--loss_mse_weight', type=float, default=0.8,
                        help='weight of the MSE term in the training loss (default: 0.8)')
    parser.add_argument('--loss_temporal_weight', type=float, default=0.1,
                        help='weight of the temporal-correlation term (default: 0.1)')
    parser.add_argument('--loss_roi_weight', type=float, default=0.1,
                        help='weight of the ROI-wise-correlation term (default: 0.1)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # * Weight initialization (see initialize_encoder)
    parser.add_argument('--pretrained_weights', default='./checkpoints/labram-base.pth',
                        help="weights to initialize from before training: the LaBraM checkpoint by default "
                             "(initializes the spatiotemporal encoder), or a UnEBOLT checkpoint when "
                             "--train_test_mode full_retainvu. Empty string ('') trains from scratch.")
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='student.', type=str,
                        help="if non-empty, strip the LaBraM 'student.' prefix when loading the "
                             "pretrained checkpoint; pass '' to keep keys as-is")
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters

    parser.add_argument('--output_dir', default='./checkpoints/run1/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./log/unebolt/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--resume', '--ckpt', default='',
                        help='checkpoint to load: resume training, or the model to evaluate with inference.py')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=False)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # ---- Dataset params ----
    parser.add_argument(
        "--atlas", type=str, default="Difumo", help="atlas used for extracting roi"
    )
    parser.add_argument('--nb_roi', default=64, type=int,
                        help='number of the output ROI (model prediction heads)')
    parser.add_argument('--difumo_dim', default=None, type=int,
                        help='DiFuMo source resolution to load (the fMRI_difumo{difumo_dim} folder). '
                             'Defaults to --nb_roi, so full-brain runs only need --nb_roi. Set it '
                             'explicitly only when predicting a subset (e.g. a single ROI) of a '
                             'higher-resolution atlas, e.g. --difumo_dim 64 --nb_roi 1 --labels_roi Thalamus.')
    parser.add_argument(
        "--labels_roi", type=str, default='full64',
        help="fMRI ROI selection: 'full64' keeps all 64 DiFuMo ROIs, or a single ROI name from the atlas"
    )
    parser.add_argument('--dataset', default='VU', type=str,
                        help='dataset')
    parser.add_argument('--dataset_root', default='./data', type=str,
                        help='root folder for dataset (HuggingFace NeurdyLab/NeuroBOLT is auto-downloaded here)')
    parser.add_argument('--prepro_datapath', default=None, type=str,
                        help='optional path to a preprocessed (train, val, test, args) pickle; '
                             'if given it is loaded directly and skips the HuggingFace build path')
    parser.add_argument('--train_test_mode', default='interscan', type=str,
                        help='intrascan/interscan/full_retainvu/')
    parser.add_argument('--split_file', default='./scan_split.yaml', type=str,
                        help='Path to the YAML file defining the train/val/test scan split for cross-subject training')
    parser.add_argument('--eval_split', default='test', type=str,
                        help="which split (train/val/test) inference.py evaluates on")
    parser.add_argument('--no_figures', action='store_true', default=False,
                        help="inference.py: skip saving per-scan FC heatmap figures")

    # ---- Dataset specific info ----
    parser.add_argument('--TR', default=2.1, type=float,
                        help='the TR value (Repetition Time) for fMRI collection in seconds')
    parser.add_argument('--mri_sync_event', default='R149', type=str,
                        help='the event name in EEG data to indicate the time for MRI scan collection')
    parser.add_argument('--nb_eegchan', default=26, type=int,
                        help='Number of EEG channels when training from scratch. If fine-tuning a model with '
                             'pre-trained weights, ensure the channel number matches the original model.')
    parser.add_argument(
        "--dataname", type=str, default="vcon26-scan01", help="data name with sub and scan for loading data for "
                                                             "intra-scan training ")

    # ---- Preprocessing config ----
    parser.add_argument('--ifremove_glb', default=False,
                        help='If regress out global signal in fMRI ROI time series')
    parser.add_argument('--ifremove_physio', default=False,
                        help='If regress out physio signal in fMRI ROI time series')

    args = parser.parse_args()
    # --difumo_dim follows --nb_roi unless the user overrides it (e.g. subset prediction).
    if args.difumo_dim is None:
        args.difumo_dim = args.nb_roi
    return args


def get_models(args):
    if args.model != "unebolt":
        raise ValueError(f"Unknown model '{args.model}'. Supported: 'unebolt'.")
    model = create_model(
        args.model,
        EEG_channel=args.nb_eegchan,
        num_roi=args.nb_roi,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
    )
    return model


def initialize_encoder(model, args):
    """Initialize the model weights (from --pretrained_weights) before training.

    The source depends on --train_test_mode:
      * default (intrascan / interscan): initialize the spatiotemporal encoder from
        the pretrained LaBraM checkpoint. LaBraM stores its weights under a
        'student.' prefix, which is stripped here.
      * full_retainvu: load a full UnEBOLT checkpoint (e.g. one trained on the VU
        dataset) to transfer its weights to a new dataset that uses a subset of the
        VU channels.

    Pass --pretrained_weights '' to train from scratch.
    """
    if not args.pretrained_weights:
        print("No pretrained weights given; training from scratch.")
        return

    if args.pretrained_weights.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.pretrained_weights, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
    print("Initializing weights from %s" % args.pretrained_weights)

    if args.train_test_mode == 'full_retainvu':
        # full UnEBOLT checkpoint saved by this repo: weights live under 'model'
        checkpoint_model = checkpoint.get('model', checkpoint)
    else:
        # LaBraM base: select the state dict by model_key, then strip the 'student.' prefix
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if args.model_filter_name != '':
            checkpoint_model = OrderedDict(
                (key[len('student.'):], value)
                for key, value in checkpoint_model.items() if key.startswith('student.'))

    # drop weights that can't be reused: a mismatched head and the relative-position buffers
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    for key in list(checkpoint_model.keys()):
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)


# EEG montage of the VU dataset: 26 scalp channels (10-20 names, uppercase), in loader order.
VU_CH_NAMES = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
               'P7', 'P8', 'FPZ', 'FZ', 'CZ', 'PZ', 'POZ', 'OZ', 'FT9', 'FT10', 'TP9', 'TP10']


def get_dataset(args):
    spec_chan_ind = None
    ch_names = []
    metrics = ["mse", "corr"]
    train_dataset = None
    test_dataset = None
    val_dataset = None
    if args.train_test_mode == 'intrascan':
        train_dataset, test_dataset, val_dataset = get_datasets.prepare_onesub_dataloader(args)

        if args.dataset == 'VU':
            ch_names = VU_CH_NAMES

    elif args.train_test_mode == 'interscan':
        train_dataset, test_dataset, val_dataset = get_datasets.prepare_full_dataloader(args)
        if args.dataset == 'VU':
            ch_names = VU_CH_NAMES

    elif args.train_test_mode == 'full_retainvu':
        train_dataset, test_dataset, val_dataset = get_datasets.prepare_full_dataloader(args)
        ch_names_full_vu = VU_CH_NAMES
        # Here, if you are using the pretrained weights of UnEBOLT trained on VU dataset, ch_names should be the
        # subset of full_vu channels. Below is an example channel set, you could modify it to adapt your own dataset
        ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
                    'P7', 'P8', 'FZ', 'CZ', 'PZ', 'OZ', 'TP9', 'TP10', 'POZ']
        spec_chan_ind = [ch_names_full_vu.index(ch) for ch in ch_names]

    if ch_names:
        return train_dataset, test_dataset, val_dataset, ch_names, metrics, spec_chan_ind
    else:
        raise ValueError(
            "Channel names are missing. Please provide channel names for your own dataset in the 'ch_names' variable.")


def main(args):
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    dataset_train, dataset_test, dataset_val, ch_names, metrics, spec_chan_ind = get_dataset(args)

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    if dataset_val is not None:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if type(dataset_test) == list:
            sampler_test = [torch.utils.data.SequentialSampler(d) for d in dataset_test]
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=CyclicSequentialBatchSampler(
            dataset_train,
            batch_size=args.batch_size,
            drop_last=False,
            shuffle=True
        ),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=False
    )

    if dataset_val is not None:
        if args.dataset == "VU" and args.train_test_mode == 'interscan':
            # One batch per scan, so val/test metrics (incl. per-scan FC) are computed
            # scan-by-scan. Batch size = samples / (number of scans in that split),
            # derived from the split file so it adapts automatically when the split changes.
            with open(args.split_file, "r", encoding="utf-8") as f:
                _split = yaml.safe_load(f)
            n_val = max(1, len(_split.get("val") or []))
            n_test = max(1, len(_split.get("test") or []))
            bs_val = max(1, len(dataset_val) // n_val)
            bs_test = max(1, len(dataset_test) // n_test)
        else:
            bs_val = len(dataset_val)
            bs_test = len(dataset_test)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=bs_val,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                # batch_size=int(1.5 * args.batch_size),
                batch_size=1,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=bs_test,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
    else:
        data_loader_val = None
        data_loader_test = None

    model = get_models(args)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    initialize_encoder(model, args)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0:
        num_layers = model_without_ddp.get_num_layers()
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()

    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    criterion = CustomMSECorrLoss(loss_weights={
        "mse": args.loss_mse_weight,
        "temporal": args.loss_temporal_weight,
        "roi": args.loss_roi_weight,
    })
    print("criterion = %s, loss_weights = %s" % (type(criterion).__name__, criterion.loss_weights))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        test_mse = []
        test_corr = []
        for data_loader in data_loader_test:
            test_stats = evaluate(data_loader, model, device, criterion=criterion, header='Test:',
                                  ch_names=ch_names, metrics=metrics, is_binary=(args.nb_roi == 1))
            test_mse.append(test_stats['mse'])
            test_corr.append(test_stats['corr'])
        print(f"======MSE: {np.mean(test_mse)} {np.std(test_mse)}, corr: {np.mean(test_corr)} {np.std(test_corr)},"
              f"conn_spacorr:{test_stats['conn_spatial_corr']}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_corr_test = 0.0
    min_loss = 10
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            ch_names=ch_names, is_binary=args.nb_roi == 1, spec_chan=spec_chan_ind
        )

        if data_loader_val is not None:
            val_stats = evaluate(data_loader_val, model, device, criterion=criterion, header='Val:',
                                 ch_names=ch_names, metrics=metrics, is_binary=args.nb_roi == 1)
            print(f"Correlation of the network on the {len(dataset_val)} val EEG: {val_stats['corr']:.2f}")
            test_stats = evaluate(data_loader_test, model, device, criterion=criterion, header='Test:',
                                  ch_names=ch_names, metrics=metrics, is_binary=args.nb_roi == 1)
            print(f"Correlation of the network on the {len(dataset_test)} test EEG: {test_stats['corr']:.2f}")
            train_corr = train_stats["corr"]
            val_mse = val_stats["loss"]
            val_corr = val_stats["corr"]
            test_mse = test_stats["mse"]
            test_corr = test_stats["corr"]
            conncorr = test_stats["conn_spatial_corr"]
            epochname = f"{epoch}" + f"-trcorr{train_corr:.2f}-valloss{val_mse:.4f}-valcorr{val_corr:.4f}-testmse{test_mse:.4f}-testcorr{test_corr:.4f}-conncorr{conncorr:.4f}"
            val_conncorr = val_stats["conn_spatial_corr"]
            print(f"[epoch {epoch}] Tcorr val/test = {val_corr:.4f}/{test_corr:.4f} | "
                  f"FC-corr val/test = {val_conncorr:.4f}/{conncorr:.4f} | "
                  f"test ConnMSE={test_stats['conn_mse']:.4f} "
                  f"F1@.25/.5={test_stats['f1_025']:.3f}/{test_stats['f1_05']:.3f}")
            if args.output_dir:
                _csv_path = os.path.join(args.output_dir, "metrics.csv")
                _write_header = not os.path.exists(_csv_path)
                with open(_csv_path, mode="a", encoding="utf-8") as _cf:
                    if _write_header:
                        _cf.write("epoch,train_corr,val_tcorr,val_fc_corr,test_tcorr,test_fc_corr,"
                                  "test_conn_mse,test_f1_025,test_f1_05,val_loss,test_mse\n")
                    _cf.write(f"{epoch},{train_corr:.6f},{val_corr:.6f},{val_conncorr:.6f},"
                              f"{test_corr:.6f},{conncorr:.6f},{test_stats['conn_mse']:.6f},"
                              f"{test_stats['f1_025']:.6f},{test_stats['f1_05']:.6f},"
                              f"{val_stats['loss']:.6f},{test_mse:.6f}\n")
            if min_loss > val_stats["loss"]:
                min_loss = val_stats["loss"]
                max_corr_test = test_stats["corr"]
                if args.output_dir and args.save_ckpt and test_corr > 0:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epochname, model_ema=model_ema)

            # Periodic save: keep a checkpoint every --save_ckpt_freq epochs
            # (--save_ckpt_freq 1 = every epoch), in addition to the best-val-loss one above.
            if args.output_dir and args.save_ckpt and args.save_ckpt_freq and (epoch + 1) % args.save_ckpt_freq == 0:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epochname, model_ema=model_ema)

            print(f'Min loss val: {min_loss:.2f}, max corr test: {max_corr_test:.2f}')
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'corr':
                        log_writer.update(correlation=float(value), head="val", step=epoch)
                    elif key == 'mse':
                        log_writer.update(MSE=float(value), head="val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=float(value), head="val", step=epoch)
                    elif key == 'conn_spatial_corr':
                        log_writer.update(conn_spatial_corr=float(value), head="val", step=epoch)

                for key, value in test_stats.items():
                    if key == 'corr':
                        log_writer.update(correlation=float(value), head="test", step=epoch)
                    elif key == 'mse':
                        log_writer.update(MSE=float(value), head="test", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=float(value), head="test", step=epoch)
                    elif key == 'conn_spatial_corr':
                        log_writer.update(conn_spatial_corr=float(value), head="test", step=epoch)

            log_stats = {**{f'train_{k}': float(v) for k, v in train_stats.items()},
                         **{f'val_{k}': float(v) for k, v in val_stats.items()},
                         **{f'test_{k}': float(v) for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': float(v) for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    start_time = time.time()

    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
