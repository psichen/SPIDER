import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from models.unet import Unet
import numpy as np
from skimage import io
import os
import argparse
from utils import *
from tqdm import tqdm

class pred_net():
    def __init__(self, checkpoint_idx, bs, checkpoint_path, data_path, prediction_path, patch_size=(64,64), overlap_th=.2, qmin=0, qmax=.99, scale_factor=3):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_idx = checkpoint_idx
        self.model = Unet(filter_base=32, unet_depth=3)
        # self.model = SwinIR(img_size=63, window_size=7, upscale=1, in_chans=1, upsampler='')
        self.bs = bs
        self.data_path = data_path
        self.patch_size = patch_size
        self.overlap_th = overlap_th
        self.qmin = qmin
        self.qmax = qmax
        self.prediction_path = prediction_path
        self.scale_factor = scale_factor

    def predict(self, rank, ensemble_idx):
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        checkpoint_save_path = os.path.join(self.checkpoint_path, f"ensemble{ensemble_idx}")
        prediction_save_path = os.path.join(self.prediction_path, f"ensemble{ensemble_idx}")
        if not os.path.exists(prediction_save_path):
            os.makedirs(prediction_save_path, exist_ok=True)


        data_A, img_shape, mi, ma = load_data(self.data_path, device, patch_size=self.patch_size, overlap_th=self.overlap_th, qmin=self.qmin, qmax=self.qmax)
        dataset = imgs_sets(data_A)
        dataloader = DataLoader(dataset, batch_size=self.bs)
        outputs = []

        self.model.load_state_dict(torch.load(os.path.join(checkpoint_save_path, f"checkpoint_{self.checkpoint_idx}.pth")))
        model = self.model.to(device)
        model.eval()

        outputs = []
        with torch.no_grad():
            for inputs in dataloader:
                b,c,h,w = inputs.size()
                outputs_tmp = model(interpolate(inputs, h*self.scale_factor, w))
                outputs.append(outputs_tmp)

        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.to(torch.device('cpu'))
        outputs = outputs.numpy()
        outputs = np.squeeze(outputs)

        self.patch_size_new = (self.patch_size[0]*self.scale_factor, self.patch_size[1])
        img_shape_new = (img_shape[0], img_shape[1]*self.scale_factor, img_shape[2])
        outputs = stitch_patches(outputs, img_shape_new, self.patch_size_new, self.overlap_th, sigma=1/3)
        # outputs = outputs*(ma - mi) + mi
        #==================================================
        outputs *= ma
        outputs += mi
        #==================================================

        io.imsave(f"{prediction_save_path}/{self.checkpoint_idx}.tif", outputs.astype(np.float32), check_contrast=False)

def predict_worker(rank, ensemble_idx, opt):
    predictor = pred_net(
            checkpoint_idx = opt.checkpoint_idx,
            bs = opt.batch_size,
            checkpoint_path = opt.checkpoint_path,
            data_path = opt.data_path,
            patch_size = opt.patch_size,
            overlap_th = opt.overlap_th,
            qmin = opt.qmin,
            qmax = opt.qmax,
            prediction_path = opt.prediction_path,
            scale_factor = opt.scale_factor,
            )
    predictor.predict(rank, ensemble_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('-i', '--checkpoint_idx', type=str, default='last')
    parser.add_argument('-cp', '--checkpoint_path', type=str, default='checkpoints')
    parser.add_argument('-dp', '--data_path', type=str, default='datasets')
    parser.add_argument('-ps', '--patch_size', type=int, nargs='+', default=[64])
    parser.add_argument('-th', '--overlap_th', type=float, default=.2)
    parser.add_argument('--qmin', type=float, default=0)
    parser.add_argument('--qmax', type=float, default=.99)
    parser.add_argument('-pp', '--prediction_path', type=str, default='predictions')
    parser.add_argument('-ws', '--world_size', type=int, default=torch.cuda.device_count())
    parser.add_argument('-s', '--scale_factor', type=int, default=3)
    opt = parser.parse_args()
    assert opt.world_size <= torch.cuda.device_count()
    if len(opt.patch_size) == 1:
        opt.patch_size = (opt.patch_size[0], opt.patch_size[0])
    elif len(opt.patch_size) == 2:
        opt.patch_size = (opt.patch_size[0], opt.patch_size[1])
    else:
        raise ValueError("patch_size must be 1 or 2 integers")

    ensembles_n = len([d for d in os.listdir(opt.checkpoint_path) if d.startswith('ensemble')])
    ensembles = np.arange(ensembles_n, dtype=int)
    opt.world_size = min(opt.world_size, ensembles_n)

    mp.set_start_method('spawn')
    with tqdm(total=ensembles_n, desc='predicting', unit='ensemble') as pbar:
        for i in ensembles[::opt.world_size]:
            group_idx = ensembles[i:i+opt.world_size]

            processes = []
            for j in group_idx:
                rank = int(j%opt.world_size)
                p = mp.Process(
                        target=predict_worker,
                        args=(rank, j, opt,)
                        )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                pbar.update()
