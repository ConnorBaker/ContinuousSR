import argparse
import os

import torch
from torchvision.transforms.v2.functional import to_dtype
from torchvision.io import decode_image, write_jpeg

import models

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--output')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = decode_image(args.input, mode="RGB")
    _, height, width = img.shape
    img = to_dtype(img, dtype=torch.float32, scale=True)
    img = img.cuda()
    
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    s1, s2 = list(map(int, args.scale.split(',')))
    scale = torch.tensor([[s1, s2]]).cuda()
  
    with torch.no_grad():
        pred = model(img.unsqueeze(0), scale).squeeze(0)
        pred = pred.clamp(0,1)

    pred = to_dtype(pred, dtype=torch.uint8, scale=True)
    pred = pred.cpu() # Didn't compile torchvision with nvJPEG
    write_jpeg(pred, args.output)
    print("finished!")
