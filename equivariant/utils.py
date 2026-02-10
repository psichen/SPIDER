import torch
from torch.utils.data import Dataset
from skimage import io
import os
from einops import rearrange
import numpy as np
import torch.nn.functional as F
import math

# --------------------------------------------------
def generate_mask(b, c, h, w, scale_factor):
    mask = torch.zeros(b,c,h,w)
    mask[:,:,::scale_factor,:] = 1
    mask = mask.bool()
    return mask

def apply_mask(upscaled_imgs, mask, scale_factor):
    b,c,up_h,w = upscaled_imgs.size()
    h = int(math.ceil(up_h/scale_factor))
    imgs = upscaled_imgs[mask].reshape(b,c,h,w)
    return imgs

def interpolate(imgs, up_h, w, mode='bicubic'):
    assert imgs.dim() == 4
    upscaled_imgs = F.interpolate(imgs, size=(up_h,w), mode=mode)
    return upscaled_imgs
# ==================================================


# --------------------------------------------------
def gaussian_mask(h, w, sigma=1/3):
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    xx, yy = np.meshgrid(x, y)
    mask = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return mask.astype(np.float32)

def get_patch_index(img_size, patch_size, overlap_th=.2):
    i_beg = [0]

    if img_size <= patch_size:
        i_end = [img_size]
    else:
        i_end = [patch_size]
        min_overlap = int(patch_size*overlap_th)
        patch_num = (img_size - min_overlap) // (patch_size - min_overlap) + 1
        overlap = (patch_num*patch_size - img_size) // (patch_num - 1)
        assert overlap >= min_overlap

        residual = (patch_size-overlap) * patch_num + overlap - img_size
        overlap_seq = [overlap]*(patch_num-1)
        for i in range(residual):
            overlap_seq[-i-1] += 1

        for ol in overlap_seq:
            i_beg.append(i_end[-1]-ol)
            i_end.append(i_beg[-1]+patch_size)

    return i_beg, i_end

def tile_patches(imgs, patch_size=(64,64), overlap_th=.2):
    if imgs.ndim == 2:
        imgs = np.expand_dims(imgs, axis=0)
    assert imgs.ndim == 3
    b,h,w = imgs.shape
    patch_size_y, patch_size_x = patch_size

    y_beg, y_end = get_patch_index(h, patch_size_y, overlap_th)
    x_beg, x_end = get_patch_index(w, patch_size_x, overlap_th)

    patches = []
    for i in range(b):
        for y1,y2 in zip(y_beg, y_end):
            for x1,x2 in zip(x_beg, x_end):
                patches.append(imgs[i, y1:y2, x1:x2])
    return np.asarray(patches)

def stitch_patches(patches, image_shape, patch_size=(64,64), overlap_th=0.2, sigma=1/3):
    b,h,w = image_shape
    patch_size_y, patch_size_x = patch_size

    patch_size_x = min(patch_size_x, w)
    patch_size_y = min(patch_size_y, h)

    y_beg, y_end = get_patch_index(h, patch_size_y, overlap_th)
    x_beg, x_end = get_patch_index(w, patch_size_x, overlap_th)

    # Create Gaussian weight mask
    g_mask = gaussian_mask(patch_size_y, patch_size_x, sigma)

    reconstructed = np.zeros((b, h, w), dtype=np.float32)
    weight_map = np.zeros((b, h, w), dtype=np.float32)

    patch_idx = 0
    for i in range(b):
        for y1, y2 in zip(y_beg, y_end):
            for x1, x2 in zip(x_beg, x_end):
                patch = patches[patch_idx].astype(np.float32)
                weighted_patch = patch * g_mask
                reconstructed[i, y1:y2, x1:x2] += weighted_patch
                weight_map[i, y1:y2, x1:x2] += g_mask
                patch_idx += 1

    reconstructed /= np.maximum(weight_map, 1e-8)
    return np.squeeze(reconstructed)

class imgs_sets(Dataset):
    def __init__(self, inputs_A):
        self.inputs_A = inputs_A

    def __len__(self):
        return len(self.inputs_A)

    def __getitem__(self, idx):
        return self.inputs_A[idx]

def load_data(data_path, device, aug=False, qmin=0, qmax=.99, patch_size=(64,64), overlap_th=.2):
    imgs_lr = io.imread(os.path.join(data_path, 'input.tif'))
    if imgs_lr.ndim == 2: # hw
        imgs_lr = rearrange(imgs_lr, 'h w -> 1 h w')
    
    # for image restoration
    #==================================================
    #imgs_lr = imgs_lr[:64] # use a subset for training, remove this line for full data
    #==================================================
    
    img_shape = imgs_lr.shape

    # normalization by quantiles
    # mi, ma = np.quantile(imgs_lr, (qmin, qmax))
    # imgs_lr = np.clip((imgs_lr - mi)/(ma - mi + 1e-20), 0, 1)
    #==================================================
    imgs_mean = np.mean(imgs_lr, axis=(-1,-2), keepdims=True)
    imgs_std = np.std(imgs_lr, axis=(-1,-2), keepdims=True)
    imgs_lr -= imgs_mean
    imgs_lr /= imgs_std
    #==================================================
    imgs_lr = tile_patches(imgs_lr, patch_size, overlap_th)

    if aug:
        imgs_lr_rot = np.rot90(imgs_lr, 2, axes=(-2,-1))
        imgs_lr = np.concatenate((imgs_lr, imgs_lr_rot))

    if imgs_lr.ndim == 3: # bhw
        imgs_lr = rearrange(imgs_lr, 'b h w -> b 1 h w')

    imgs_lr = torch.from_numpy(imgs_lr).to(device, dtype=torch.float32)

    # return imgs_lr, img_shape, mi, ma
    return imgs_lr, img_shape, imgs_mean, imgs_std
# ==================================================