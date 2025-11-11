import pandas as pd
import numpy as np
import utils
import os
import torch
import argparse
import sys
import torch.optim as optim
import torch.nn.functional as F
import itertools
import controlnet.model as cmodel
import controlnet.utils as utils
import warnings
import time
from tabsyn.latent_utils import get_input_train
from tabsyn.model import MLPDiffusion, Model
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pdb


warnings.filterwarnings('ignore')


def set_anneal_lr(opt, init_lr, step, all_steps):
    frac_done = step / all_steps
    lr = init_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr

def controlnet_training(train_x, cf, cdim, diffuser, diffuserx, save_path, device, d_hidden=[512, 1024, 1024, 512], steps=1000, lr=0.0001, drop_out=0.0, bs=1024, add_noise=False, noise_scale=10, train_minority = False, noisetype='laplace', weight=1):

    train_data = diffuserx

    print(f'noise type is {noisetype}')

    if not isinstance(train_x, torch.Tensor):
        train_x = torch.from_numpy(train_x).float()
        cf = torch.from_numpy(cf).float()
    device = torch.device('cuda:0')
    model = cmodel.ControlRNet(train_x.shape[1], cdim, d_hidden, drop_out)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    r = torch.randperm(train_x.shape[0])
    ds = [train_x[r]]
    ds2 = [cf[r],train_data[r]]
    dl = utils.prepare_fast_dataloader(ds, batch_size = bs, shuffle = True, moretensors = ds2)
    model.train()
    model.to(device)
    diffuser.to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.00001)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=20, verbose=False)
    sta = time.time()
    idx=0
    curr_count = 0
    curr_loss = 0

    while idx < steps:
        x = next(dl)[0]
        x = x.to(device)
        xf = next(dl)[1]
        x_diffuser = next(dl)[2]
        if add_noise:
            if noisetype == 'laplace':
                xf = utils.tensor_add_laplace_noise(xf, noise_scale)
            elif noisetype == 'uniform':
                xf = utils.tensor_add_uniform_noise(xf, noise_scale)
            else:
                xf = utils.tensor_add_normal_noise(xf, noise_scale)
        #pdb.set_trace()
        x_diffuser = x_diffuser.to(device)
        xf = xf.to(device)
        cond = None
        opt.zero_grad()
        b = x.shape[0]


        loss = diffuser(x_diffuser, controlnet=True, model = model, xf = xf.float(), weight=weight)
        loss = loss.mean()

        scheduler.step(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        set_anneal_lr(opt, lr, idx, steps)

        curr_count += len(x)
        curr_loss += loss.item() * len(x)
        if (idx + 1) % 100 == 0:
            loss = np.around(curr_loss/ curr_count, 4)
            if (idx + 1) % 500 == 0:
                print(f'Step {(idx + 1)}/{steps} Loss: {loss}')
            curr_count = 0
            curr_loss = 0.0
        idx += 1
    train_end = time.time()
    print(f'training time: {train_end-sta}')

    model.to(torch.device('cpu'))
    model.eval()
    torch.save(model, save_path)

def main(args):

    save_dir = args.dataname
    savename = args.save_name
    noisetype = args.noise

    device = torch.device(f'cuda:{args.device}')
    scale = args.scale
    print(f'scale is {scale}')
    config = utils.load_json(f'data/Info/{args.dataname}.json')

    #load diffuser
    train_z, _, _, ckpt_path, _ = get_input_train(args)
    in_dim = train_z.shape[1] 
    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    diffuser = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)
    diffuser.load_state_dict(torch.load(f'tabsyn/ckpt/{args.dataname}/model.pt'))
    for param in diffuser.parameters():
        param.requires_grad = False

    controlnet_training(train_x = train_z,
                        cf = train_z, #this is condition
                        cdim = train_z.shape[1],
                        diffuser = diffuser,
                        diffuserx = train_z, # this is the input of diffusion
                        save_path = f'{ckpt_path}/{savename}.pt', 
                        device=device, 
                        d_hidden=args.diffuser_dim, 
                        steps=args.diffuser_steps, 
                        lr=0.0018, 
                        drop_out=0.0, 
                        bs=args.diffuser_bs,
                        add_noise=args.addnoise,
                        noise_scale=scale,
                        train_minority = args.diffuser_minority,
                        noisetype = noisetype,
                        weight = args.weight
                        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of controlnet')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')

    parser.add_argument('--dataset-name', type=str, default='default')

    parser.add_argument('--diffuser-dim', nargs='+', type=int, default=(512, 1024, 1024, 512))
    parser.add_argument('--diffuser-lr', type=float, default=0.005)
    parser.add_argument('--diffuser-steps', type=int, default=30000)
    parser.add_argument('--diffuser-bs', type=int, default=4096)
    parser.add_argument('--diffuser-timesteps', type=int, default=1000)
    parser.add_argument('--diffuser-minority', type=bool, default=False)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--addnoise', type=bool, default=True)
    parser.add_argument('--weight', type=float, default=1.0)

    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--scale-factor', type=float, default=8.0)
    parser.add_argument('--save_name', type=str, default='output')
    parser.add_argument('--noise', type=str, default='laplace')


    args = parser.parse_args()
    main(args)