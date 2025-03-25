# Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling
> **Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling**<br>
Long Peng<sup>1,3</sup>†, Anran Wu<sup>1,2</sup>†, Wenbo Li<sup>3</sup>*, Peizhe Xia<sup>1</sup>, Xueyuan Dai<sup>4</sup>, Xinjie Zhang<sup>5</sup>, Xin Di<sup>1</sup>, Haoze Sun<sup>6</sup>, Renjing Pei<sup>3</sup>, Yang Wang<sup>1,4*</sup>, Yang Cao<sup>1</sup>, Zheng-Jun Zha<sup>1</sup>. † Equal Contribution. * Corresponding Authors.

> <sup>1</sup>USTC, <sup>2</sup>AHU, <sup>3</sup>Huawei Noah’s Ark Lab, <sup>4</sup>Chang’an University, <sup>5</sup>HKUST, <sup>6</sup>THU. 

[[Arxiv Paper](https://arxiv.org/pdf/2503.06617)]&nbsp;
[[Website Page](https://github.com/peylnog/ContinuousSR)]&nbsp;
[[Google Drive](...)]&nbsp;
[[Colab Notebook](...)]&nbsp;

## :rocket:  :rocket:  :rocket: **News**
- [x]  **Arxiv Version has been released.**
- [x]  **Test code has been released.**
- [x]  **Pretrained model has been released.**
- [x]  Since this paper is completed within the company, the related code and pre-trained models cannot be released. To support the open-source community, we voluntarily and freely attempt to reproduce it in this repository. The reproduced results may have very slight deviations, but the difference stays within 0.05 dB.

## TODO
- [ ] Release train code.

## Framework
![method](assets/motivation.png)&nbsp;
Compared to other methods, the proposed ContinuousSR delivers significant improvements in SR quality across all scales, with an impressive 19.5× speedup when continuously upsampling an image across forty scales.

## Summary
We introduce **ContinuousSR**, a novel Pixel-to-Gaussian paradigm designed for fast and high-quality arbitrary-scale super-resolution. By explicitly reconstructing 2D continuous HR signals from LR images using Gaussian Splatting, **ContinuousSR** significantly improves both efficiency and performance. Through statistical analysis, we uncover the Deep Gaussian Prior (DGP) and propose a DGP-driven Covariance Weighting mechanism along with an Adaptive Position Drifting strategy. These innovations improve the quality and fidelity of the reconstructed Gaussian fields. Experiments on seven popular benchmarks demonstrate that our method outperforms state-of-the-art methods in both quality and speed, achieving a {19.5×} speed improvement and {0.90dB} PSNR improvement, making it a promising solution for ASSR tasks.

### Dependencies and Installation
- python=3.9
- pytorch=1.13
- basicsr==1.3.4.9
- Others:

```bash
git clone https://github.com/XingtongGe/gsplat.git
cd gsplat
pip install -e .[dev]
```

## Get Started
### Pretrained model
The pretrained model can be downloaded from [pretrained model](https://drive.google.com/file/d/1UKXch2ryl6zZWs9QCgtfWpgVwtYLsxtS/view?usp=drive_link). After downloading, place the model in the designated folder, and you can then proceed with the following demo and inference operations.

### Demo
Here is an Demo
```bash
# scale represents the magnification factors for height and width respectively
python demo.py --input butterflyx4.png --model ContinuousSR.pth --scale 4,4 --output output.png
```
### Inference
Here is an example command for inference
```bash
# test Set5 X4
python test.py --config ./configs/test/test-set5-4.yaml --model ContinuousSR.pth
```

## Visual Examples
![method](assets/vis1.png)&nbsp;

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

## Acknowledgement
This repository is maintained by [Long Peng](https://peylnog.github.io/) and [Anran Wu](https://github.com/wuanran678).

### Citation

If you are interested in the following work, please cite the following paper.

```
@article{peng2025pixel,
  title={Pixel to Gaussian: Ultra-Fast Continuous Super-Resolution with 2D Gaussian Modeling},
  author={Peng, Long and Wu, Anran and Li, Wenbo and Xia, Peizhe and Dai, Xueyuan and Zhang, Xinjie and Di, Xin and Sun, Haoze and Pei, Renjing and Wang, Yang and others},
  journal={arXiv preprint arXiv:2503.06617},
  year={2025}
}
```
