# SPIDER
self-supervised frameework for Scanning Probe microscopy Image DEnoising and Restoration

## Installation
tested platform: Rocky Linux 9.1, 256G RAM, NVIDIA TITAN X 12GB
- Python = 3.9.14
- CUDA = 12.0

`pip install -r requirements.txt`

Usually it would take couple of minutes to finish installtion. Instructions to use SPIDER can be found in the `SPIDER.ipynb`. For typical HS-AFM images (over tens of frames of 300 by 300 pixels), it would take around 5 minutes to perform ensemble denoising results. 
