import argparse
import os

import torch
from torchvision.transforms.v2.functional import to_dtype, resize
from torchvision.io import decode_image, write_jpeg
from torch.profiler import profile, ProfilerActivity, record_function
import numpy as np

import models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--model")
    parser.add_argument("--scale")
    parser.add_argument("--output")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision("medium")

    img = decode_image(args.input, mode="RGB").cuda()
    _, height, width = img.shape
    
    with torch.device('cuda'):
        img = to_dtype(img, dtype=torch.bfloat16, scale=True)
        img = resize(img, [height // 4, width // 4])

        model_spec = torch.load(args.model)["model"]
        model = models.make(model_spec, load_sd=True).to(dtype=torch.bfloat16)
        model = torch.compile(model, mode="max-autotune")
        s1, s2 = list(map(int, args.scale.split(",")))
        scale = np.array([[s1, s2]])

        torch.cuda.memory._record_memory_history()
        with (
            torch.no_grad()
            # profile(
            #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #     profile_memory=True,
            #     # record_shapes=True,
            #     # with_stack=True,
            #     # with_flops=True,
            # ) as prof,
        ):
            pred = model(img.unsqueeze(0), scale).squeeze(0)
            pred = pred.clamp(0, 1)

        torch.cuda.memory._dump_snapshot("memory-snapshot-4-timm-window-autotune.pickle")
        torch.cuda.memory._record_memory_history(None)

        # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total"))
        # print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
        # print(prof.key_averages().table(sort_by="cuda_memory_usage"))
        # prof.export_chrome_trace("chrome.json")
        # prof.export_memory_timeline("memory-div-32.html")

        pred = to_dtype(pred, dtype=torch.uint8, scale=True)
    pred = pred.cpu() # Didn't compile torchvision with nvJPEG
    write_jpeg(pred, args.output)
    print("finished!")
