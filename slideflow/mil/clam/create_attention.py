"""Modification of https://github.com/mahmoodlab/CLAM"""

import argparse
import csv
import os
import pdb

import numpy as np
import torch
import torch.nn.functional as F

from slideflow import log
from slideflow.mil.models import CLAM_MB, CLAM_SB
from .utils import *
from .utils.eval_utils import initiate_model as initiate_model


def path_to_name(path):
    '''Returns name of a file, without extension, from a given full path string.'''
    _file = path.split('/')[-1]
    if len(_file.split('.')) == 1:
        return _file
    else:
        return '.'.join(_file.split('.')[:-1])

def path_to_ext(path):
    '''Returns extension of a file path string.'''
    _file = path.split('/')[-1]
    if len(_file.split('.')) == 1:
        return ''
    else:
        return _file.split('.')[-1]

def infer_single_slide(model, features, label, reverse_label_dict, k=1, silent=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = features.to(device)
    with torch.no_grad():
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            logits, A, _ = model(features, return_attention=True)
            Y_hat = torch.topk(logits, 1, dim=1)[1].item()
            Y_prob = F.softmax(logits, dim=1)

            if isinstance(model, (CLAM_MB,)):
                A = A[Y_hat]

            A = A.view(-1, 1).cpu().numpy()

        else:
            raise NotImplementedError

        if not silent:
            log.info('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))

        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, A

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params

def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict

def export_attention(model_args, ckpt_path, outdir, pt_files, slides, reverse_labels, labels): #Should include n_classes, model_size):

    '''
    reverse_label_dict = {
        0: "LOW",
        1: "HIGH"
        ... etc
    }
    '''

    model_args = argparse.Namespace(**model_args)

    # Load model
    log.info(f'Exporting attention using checkpoint {ckpt_path}')
    model =  initiate_model(model_args, ckpt_path)

    # load features
    logits = {s:{'pred':None,'prob':None} for s in slides}
    log.info(f"Working on {len(slides)} slides.")

    for slide in slides:
        csv_save_loc = os.path.join(outdir, f'{slide}.csv')
        features_path = os.path.join(pt_files, f'{slide}.pt')
        features = torch.load(features_path)


        Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, labels[slide], reverse_labels, model_args.n_classes, silent=True)

        logits[slide]['pred'] = Y_hats_str[0]
        logits[slide]['prob'] = Y_probs[0]

        with open(csv_save_loc, 'w') as csv_file:
            writer = csv.writer(csv_file)
            for i, a in enumerate(A):
                writer.writerow([i, a[0]])
        log.info(f'Exported attention scores to [green]{csv_save_loc}[/]')
        del features

    log.info("\nSlide\tPred\tProb")
    for slide in slides:
        log.info(f"{slide}\t{logits[slide]['pred']}\t{logits[slide]['prob']}")

    with open(os.path.join(outdir, 'pred_summary.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        header = ['slide', 'prediction', 'probability']
        writer.writerow(header)
        for slide in slides:
            writer.writerow([slide, logits[slide]['pred'], logits[slide]['prob']])
    log.info(f'Exported prediction summary to {os.path.join(outdir, "pred_summary.csv")}')