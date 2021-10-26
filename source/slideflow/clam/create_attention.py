from __future__ import print_function
import numpy as np
import argparse
import torch
import pdb
import os
from slideflow.clam.utils import *
from slideflow.clam.utils.eval_utils import initiate_model as initiate_model
from slideflow.clam.models.model_clam import CLAM_MB, CLAM_SB
import csv

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
    features = features.to(device)
    with torch.no_grad():
        if isinstance(model, (CLAM_SB,)):
            logits, Y_prob, Y_hat, A, _ = model(features)
            Y_hat = Y_hat.item()

            if isinstance(model, (CLAM_MB,)):
                A = A[Y_hat]

            A = A.view(-1, 1).cpu().numpy()

        else:
            raise NotImplementedError

        if not silent:
            print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))

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
    print('\ninitializing CLAM model from checkpoint')
    print('\nckpt path: {}'.format(ckpt_path))
    model =  initiate_model(model_args, ckpt_path)

    # load features
    logits = {s:{'pred':None,'prob':None} for s in slides}
    print(f"Working on {len(slides)} slides.")

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
        print(f'Exported attention scores to {csv_save_loc}')
        del features

    print("\nSlide\tPred\tProb")
    for slide in slides:
        print(f"{slide}\t{logits[slide]['pred']}\t{logits[slide]['prob']}")

    with open(os.path.join(outdir, 'pred_summary.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        header = ['slide', 'prediction', 'probability']
        writer.writerow(header)
        for slide in slides:
            writer.writerow([slide, logits[slide]['pred'], logits[slide]['prob']])
    print(f'Exported prediction summary to {os.path.join(outdir, "pred_summary.csv")}')