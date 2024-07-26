import subprocess
import tempfile

import numpy as np
import torch
import random
from pathlib import Path
#from dataset import ProteinDataset, PaddingCollate
#from score_sde_pytorch.utils import recursive_to
from biotite.structure.io import load_structure, save_structure
import biotite.structure as struc
import shutil


def random_mask_batch(batch,device,mask_max_len=0.95,mask_min_len=0.05,random_mask_prob=0.33,contiguous_mask_prob=0.33): # 蛋白片段 mask

    B, _, N, _ = batch["coords_6d"].shape
    mask_min = mask_min_len # 0.05
    mask_max = mask_max_len # 0.95

    random_mask_prob = random_mask_prob # 0.33
    contiguous_mask_prob = contiguous_mask_prob # 0.33

    lengths = [len([a for a in i if a != "_"]) for i in batch["aa_str"]]  # get lengths without padding token
    # Decide between none vs random masking vs contiguous masking
    prob = random.random()
    if prob < random_mask_prob: # batch随机length mask 0.33 [B,L], 一段蛋白mask
        # Random masking
        mask = []
        for l in lengths:
            rand = torch.randint(int(mask_min * l), int(mask_max * l), (1,))[0] # 0.05-0.95 长度中随机取一段 rand, 100
            rand_indices = torch.randperm(l)[:rand] # 0-100

            m = torch.zeros(N)
            m[rand_indices] = 1

            mask.append(m)
        mask = torch.stack(mask, dim=0)
    elif prob > 1-contiguous_mask_prob: # 0.67
        # Contiguous masking
        mask = []
        for l in lengths:
            rand = torch.randint(int(mask_min * l), int(mask_max * l), (1,))[0] # 100
            index = torch.randint(0, (l - rand).int(), (1,))[0] # 118-100=18中随机取一个数字 14

            m = torch.zeros(N)
            m[index:index + rand] = 1 # 14:114 = 1

            mask.append(m)
        mask = torch.stack(mask, dim=0)
    else:
        mask = torch.ones(B, N) # No masking

    mask_pair = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1)) # B, N -> B, N, N # 同时为true 或 同时>0 做mask
    batch["mask_random_pair"] = mask_pair.to(device=device, dtype=torch.bool)
    batch["mask_random"] = mask.to(device=device,dtype=torch.long)

    return batch

#########################################################

import torch

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].optimizer.load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['step'] = loaded_state['step']
    #state={'optimizer':loaded_state['optimizer'],'model':loaded_state['model'],'step':loaded_state['step']}
    return state

def save_checkpoint(ckpt_dir, state):
    saved_state = {
    'optimizer': state['optimizer'].optimizer.state_dict(),
    'model': state['model'].state_dict(),
    'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)

def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        if device == 'cpu':
            return obj.cpu()
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}
    else:
        return obj
####################################################

import os
import sys

class Logger(object):
    """Writes both to file and terminal"""
    def __init__(self, savepath, mode='a'):
        self.terminal = sys.stdout
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.log = open(os.path.join(savepath, 'logfile.log'), mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
