import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import datasets
import models
import utils
from utils import make_coord
from torchvision.utils import save_image

import torch.nn.functional as F
torch.backends.cudnn.enabled = False
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'


def make_coord_and_cell(img, scale):
    scale = int(scale)
    h, w = img.shape[-2:]
    h, w = h * scale, w * scale
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    return coord.unsqueeze(0), cell.unsqueeze(0)


def batched_predict(model, inp, coord, scale, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :].contiguous(), scale.contiguous(), cell[:, ql: qr, :].contiguous())
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, scale_max=4, window_size=0):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    IDX = 1

    for batch in pbar:
        
        for k, v in batch.items():
            batch[k] = v.cuda()
            
        inp = (batch['inp'] - inp_sub) / inp_div
        gt = (batch['gt'] - gt_sub) / gt_div
        scale = batch['scale']
        scale = scale.item()
        
        lr_bicubic = F.interpolate(batch['inp'], size=batch['gt'].shape[2:], mode='bicubic', align_corners=False)
        lr_bicubic = np.squeeze(lr_bicubic, axis=0)
        
        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]
            
            coord = utils.make_coord((scale*(h_old+h_pad), scale*(w_old+w_pad))).unsqueeze(0).cuda()
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0
        
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, batch['scale'])
        else:
            pred = batched_predict(model, inp, batch['scale'], batch['cell'], eval_bsize)

        pred = pred * gt_div + gt_sub
        
        pred.clamp_(0, 1)
        save_image(pred, f'/output/experiment/output_image_{IDX}.png')

        gt = np.squeeze(batch['gt'], axis=0).cuda()
        
        save_image(gt, f'/output/experiment/gt_{IDX}.png')
        IDX = IDX+1
        
        res = metric_fn(pred, gt)

        val_res.add(res.item())

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()
                  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    parser.add_argument(
        "--num_points",
        type=int,
        default=500,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument('--BLOCK_H', default=16)
    parser.add_argument('--BLOCK_W', default=16)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=1, pin_memory=True)
    
    model_args = {"num_points": args.num_points, "BLOCK_H": args.BLOCK_H, "BLOCK_W": args.BLOCK_W}

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, model_args, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))
