import argparse
import os
import math
from functools import partial

import yaml
import torch
from torchvision import transforms
from PIL import Image

import numpy as np
import datasets
import models
import utils
from utils import make_coord
from torchvision.utils import save_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--output')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB')).cuda()
    
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    s1, s2 = list(map(int, args.scale.split(',')))
    scale = torch.tensor([[s1, s2]]).cuda()
  
    with torch.no_grad():
        pred = model(img.unsqueeze(0), scale).squeeze(0)
        pred = pred.clamp(0,1)
        
    transforms.ToPILImage()(pred).save(args.output)
    print("finished!")
