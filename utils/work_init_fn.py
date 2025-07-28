import numpy as np
import torch
import options
import random
import argparse
opt = options.Options().init(argparse.ArgumentParser(description='ShadowRemoval')).parse_args()

def worker_init_fn(worker_id):
    seed = opt.seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)