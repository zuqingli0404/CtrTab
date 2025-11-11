import numpy as np
import pandas as pd
import utils
import torch
import argparse
import sys
import controlnet.model as cmodel
import controlnet.utils as utils
import warnings
import time
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.model import MLPDiffusion, Model
from tabsyn.diffusion_utils import sample

warnings.filterwarnings('ignore')


def sampling(n_samples, sample_dim, diffuser, controlnet, cf, controlnet2, cf2, device, ifminority, weight = 1):
    b = n_samples
    cfx = cf.shape[0]
    if b <= cfx:
        cf = cf[:b]
    else:
        length = np.repeat(cf, b/cfx,axis=0).shape[0]
        cf = np.concatenate((np.repeat(cf,b/cfx,axis=0),cf[:b-length] ),axis=0)
    cf = torch.from_numpy(cf).to(device).float()
    Lt_sqrt = torch.sqrt(torch.zeros(b) + 1e-10) + 0.0001
    Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
    pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)

    controlnet.to(device)
    if cf is not None and cf2 is not None:
        print('sample with double controlnet')
        controlnet.to(device)
        controlnet2.to(device)
        samples = diffuser.sample_with_controlnet(n_samples, controlnet=controlnet, cf=cf, controlnet2=controlnet2, cf2=cf2)
    elif cf is not None:
        print('sample with single controlnet')
        controlnet.to(device)
        samples = sample(diffuser.denoise_fn_D, n_samples, sample_dim, controlnet = True, model = controlnet, cf = cf, weight=weight)
    else:
        print('sample without controlnet')
        samples = diffuser.sample(n_samples)
    return samples

def main(args):
    
    dataname = args.dataname
    device = args.device
    model_name = args.model_name
    print(f'model name is {model_name}')
    config = utils.load_json(f'data/Info/{dataname}.json')
    scale = args.scale
    weight = args.weight
    noisetype = args.noise
    
    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1] 

    mean = train_z.mean(0)
    train_z = (train_z - mean) / 2
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    
    diffuser = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    diffuser.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))

    try:
        controlnet_addr = f'./tabsyn/ckpt/{args.dataname}/{model_name}.pt'
        controlmodels = torch.load(controlnet_addr)
        controlmodels.eval()
        for param in controlmodels.parameters():
            param.requires_grad = False
    except FileNotFoundError:
        print(f'{controlnet_addr} not found!')

    start_time = time.time()

    train_x = train_z.numpy()
    if args.diffuser_minority:
        minlabel_idx = train_x[:, -1] == float(config['minority_classes'][0])
        train_minlabel = train_x[minlabel_idx]
        num_samples = train_x.shape[0]
        train_wy = train_minlabel[:, :-1][:num_samples]
        if config['minority_classes'][0] == 1:
            labelsy = np.ones((train_minlabel.shape[0], 1)).astype(float)
        else:
            labelsy = np.zeros((train_minlabel.shape[0], 1)).astype(float)
        noisy_data = np.concatenate((utils.add_laplace_noise(train_wy, scale),labelsy), axis = 1)
    else:
        if args.addnoise:
            if noisetype == 'laplace':
                print('add laplace noise')
                noisy_data = utils.array_add_laplace_noise(train_x, scale)
            elif noisetype == 'uniform':
                print('add uniform noise')
                noisy_data = utils.array_add_uniform_noise(train_x, scale)
            else:
                print('add normal noise')
                noisy_data = utils.array_add_normal_noise(train_x, scale)
        else:
            print('no noise data without minority')
            noisy_data = train_x
    try:
        n_samples = config['n_samples'][0]
    except:
        n_samples = train_x.shape[0]
    x_next = sampling(n_samples, in_dim, diffuser, controlmodels, noisy_data, None, None, device, args.diffuser_minority, weight)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    save_path = f'synthetic/{args.dataname}/{args.save_name}.csv'
    syn_df.to_csv(save_path, index = False)
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of controlnet')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')

    parser.add_argument('--dataset-name', type=str, default='default')

    parser.add_argument('--diffuser-dim', nargs='+', type=int, default=(512, 1024, 1024, 512))
    parser.add_argument('--diffuser-lr', type=float, default=0.0018)
    parser.add_argument('--diffuser-steps', type=int, default=30000)
    parser.add_argument('--diffuser-bs', type=int, default=4096)
    parser.add_argument('--diffuser-timesteps', type=int, default=1000)
    parser.add_argument('--diffuser-minority', type=bool, default=False)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--weight', type=float, default=1)
    parser.add_argument('--addnoise', type=bool, default=True)
    parser.add_argument('--noise', type=str, default='laplace')


    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--scale-factor', type=float, default=8.0)
    parser.add_argument('--save-name', type=str, default='CtrTab')
    parser.add_argument('--model-name', type=str, default='CtrTab')


    args = parser.parse_args()
    main(args)
