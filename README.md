<p>
  <h1 align="center">SPIDER</h1>
  <h3 align="center">Scanning Probe microscopy Image DEnoising and Restoration</h3>
</p>

![Logo](imgs/logo.png)

## 👀 Introduction
SPIDER is a self-supervised framework for denoising atomic force microscopy (AFM) and other scanning probe microscopy (SPM) images using paired trace and retrace scans. The method utilizes the spatial redundancy of raster-scanning and learns the underlying surface signals while suppressing independent noises  without requiring clean ground-truth images.

## 🛠 Installation
We provide a Google Colab notebook for easy use. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/psichen/SPIDER/blob/main/SPIDER_colab.ipynb)

### Tested platform
- Rocky Linux 9.1 (256G RAM, NVIDIA TITAN X 12GB)
- Python = 3.9.14
- CUDA = 12.0

### Instructions

1. Clone the repository:
```bash
git clone https://github.com/psichen/SPIDER.git
cd SPIDER
```

2. Set up the environment and package dependencies:
```bash
pip install -r requirements.txt
```

Usually it would take couple of minutes to finish installtion. For typical HS-AFM images (over tens of frames of 300 by 300 pixels), it would take around 5 minutes to perform ensemble denoising results. 

## ⚙️ Dataset preparation
First, hysteresis effect is corrected for AFM trace/retrace images to generate image pairs for self-supervised denoising:

```bash
python preprocess.py [OPTIONS]
```

Some important arguments are listed:

| Argument | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--raw_path` | `-rp` | str | 'raw' | path to raw images |
| `--trace_file` | `-tf` | str | 'trace.tif' | trace filename |
| `--retrace_file` | `-rf` | str | 'retrace.tif' | retrace filename |
| `--data_path` | `-dp` | str | 'datasets' | path to training datasets |
| `--shift` | `-sf` | float | None | attention area shift |
| `--window` | `-w` | float | .1 | attention window size |
| `--trace_left` | `-tl` | float | .05 | trace left boundary ratio |
| `--trace_right` | `-tr` | float | .95 | trace right boundary ratio |
| `--coef` | `-c` | float[3] | | predefined quadratic coefficients |
| `--plot` | `-p` | flag | | plot results if present |
| `--save` | `-s` | flag | | save results in the `data_path` if present |

The mismatch between trace columns and retrace columns is non-linear and fitted by a quadratic function. The script will give the x- and y-coordiate of the apex of the quadratic curve, representing the linear component of the mismatch. The quadratic parameters will be given too for the non-linear mismatch correction.

Example:
```bash
python preprocess.py \
    -rp ./raw \
    -dp ./datasets \
    -tf trace.tif \
    -rf retrace.tif \
    -sf .04 \
    -p -s \
```
or
```bash
python preprocess.py \
    -c 0.0009050942202696902 -0.0936764923029398 -3.1469189098574337 \
    -p \
```

A successful fitting result would be like
![Logo](imgs/hysteresis.png)

The left panel shows the displacement of retrace columns to the most similar trace columns. The quadratic curve approximates the non-linearity to the 2nd-order of the Taylor series.

Because the hysteresis effect is localized, *i.e.*, the trace columns and the retrace columns corresponding to the same scanned region are restricted in a limited range in the formed image. The middle panel shows the attention distribution between trace and retrace when the locality is considered.

The right panel shows the global attention distribution. When images experience an extremely low signal-to-noise ratio (SNR), the fitting process from gloal attention might be affected by the broad attention spread resulting from self-similar noises.

Corrected trace/retrace image pairs at sub-pixel level will be saved in the `data_path` for the following self-supervised training.

## 💻 Training
Two models are trained independently in the script to denoise trace based on retrace (r2t) and denoise retrace based on trace (t2r).

```bash
python trainer_ddp.py [OPTIONS]
```
SPIDER supports multi-GPU training using PyTorch Distributed Data Parallel (DDP).

Some important arguments are listed:

| Argument | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--data_path` | `-dp` | str | 'datasets' | path to training datasets |
| `--checkpoint_path` | `-cp` | str | 'checkpoints' | path to checkpoints |
| `--batch_size` | `-b` | int | 512 | number of images processed in each iteration |
| `--epochs` | `-e` | int | None | total number of training epochs |
| `--iteration` | `-it` | int | 200 | total number of training iterations |
| `--augmentation` | `-a` | flag | | perform data augmentation if present |
| `--patch_size` | `-ps` | int[1 or 2] | 64 | image patch size for training |
| `--ensembles` | `-n` | int | 3 | number of emsemble learnings |
| `--world_size` | `-ws` | int | `torch.cuda.device_count()` | number of GPUs |

The training script will generate `hyperparams.txt`, `.pth` checkpoint, updated learning rates and loss values in the `checkpoint_path`.

## 💡 Prediction

## 🔖 Citations
