import torch
from torch.utils.data import Dataset
from skimage import io
from skimage.metrics import structural_similarity
import os
from einops import rearrange
import numpy as np
from skimage.registration import optical_flow_tvl1
# import open3d as o3d

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
    def __init__(self, inputs_A, inputs_B):
        self.inputs_A = inputs_A
        self.inputs_B = inputs_B

    def __len__(self):
        assert len(self.inputs_A) == len(self.inputs_B)
        return len(self.inputs_A)

    def __getitem__(self, idx):
        return self.inputs_A[idx], self.inputs_B[idx]

def load_data(data_path, mode, device, aug=False, qmin=0, qmax=.99, patch_size=(64,64), overlap_th=.2):
    trace = io.imread(os.path.join(data_path, 'trace.tif'))[:96]
    retrace = io.imread(os.path.join(data_path, 'retrace.tif'))[:96]
    if trace.ndim == 2: # hw
        trace = rearrange(trace, 'h w -> 1 h w')
        retrace = rearrange(retrace, 'h w -> 1 h w')
    img_shape = trace.shape

    # normalization by quantiles
    if mode == 't2r':
        mi, ma = np.quantile(trace, (qmin, qmax))
    elif mode == 'r2t':
        mi, ma = np.quantile(retrace, (qmin, qmax))
    else:
        raise Exception("mode should be 't2r' or 'r2t'")

    trace = np.clip((trace - mi)/(ma - mi + 1e-20), 0, 1)
    retrace = np.clip((retrace - mi)/(ma - mi + 1e-20), 0, 1)

    trace = tile_patches(trace, patch_size, overlap_th)
    retrace = tile_patches(retrace, patch_size, overlap_th)

    if aug:
        trace_rot = np.rot90(trace, 2, axes=(-2,-1))
        retrace_rot = np.rot90(retrace, 2, axes=(-2,-1))
        trace = np.concatenate((trace, retrace_rot))
        retrace = np.concatenate((retrace, trace_rot))

    if trace.ndim == 3: # bhw
        trace = rearrange(trace, 'b h w -> b 1 h w')
        retrace = rearrange(retrace, 'b h w -> b 1 h w')

    trace = torch.from_numpy(trace).to(device, dtype=torch.float32)
    retrace = torch.from_numpy(retrace).to(device, dtype=torch.float32)
    
    if mode == 't2r':
        data_A = trace
        data_B = retrace
    elif mode == 'r2t':
        data_A = retrace
        data_B = trace
    else:
        raise Exception("mode should be 't2r' or 'r2t'")

    return data_A, data_B, img_shape, mi, ma
# ==================================================

# --------------------------------------------------
def pixelization(x, imgs):
    """
    average image columns if their x coordinates are within the interval of 1
    """
    ndim = imgs.ndim
    imgs_out = []
    imgs_temp = imgs[...,0] # first column of imgs
    imgs_temp = np.expand_dims(imgs_temp, 0)
    x_temp = np.floor(x[0])

    for i in range(1, len(x)):
        if np.floor(x[i]) == x_temp: # average columns within the interval 1
            imgs_temp = np.append(imgs_temp, [imgs[...,i]], axis=0)
        else:
            if imgs_temp.ndim == ndim-1:
                imgs_temp = np.expand_dims(imgs_temp, 0)
            imgs_out.append(np.mean(imgs_temp, axis=0))
            imgs_temp = imgs[...,i]
            imgs_temp = np.expand_dims(imgs_temp, 0)
            x_temp = np.floor(x[i])

    imgs_out = np.asarray(imgs_out)
    imgs_out = rearrange(imgs_out, 'w b h -> b h w')
    return imgs_out

# def pointcloud(x, y, img, down_sample=None):
    # # reverse y order
    # img = img[::-1,:]

    # x = x[::down_sample]
    # img = img[:,::down_sample]
    # assert img.ndim == 2
    # xx, yy = np.meshgrid(x,y)
    # points = [xx.flatten(), yy.flatten(), img.flatten()]
    # points = np.array(points).T
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # if down_sample is not None:
        # pcd = pcd.voxel_down_sample(down_sample)
    # return pcd
# ==================================================

# --------------------------------------------------
def imgs_optical_flow(imgs):
    b,h,w = imgs.shape
    u_list = [np.zeros((h,w))]
    v_list = [np.zeros((h,w))]
    for i in range(imgs.shape[0]):
        if i:
            u,v = optical_flow_tvl1(imgs[i-1], imgs[i])
            u_list.append(u)
            v_list.append(v)

    return u_list, v_list

def align_frames(imgs, u_list, v_list): # I[t+1,h+u,w+v] = I[t,h,w]
    u_median = np.median(u_list, axis=(-1,-2))
    v_median = np.median(v_list, axis=(-1,-2))
    u_median = np.cumsum(u_median)
    v_median = np.cumsum(v_median)
    u_median = np.rint(u_median).astype(int)
    v_median = np.rint(v_median).astype(int)

    b,h,w = imgs.shape
    imgs_pad = []
    u_pad = np.max(np.abs(u_median))
    v_pad = np.max(np.abs(v_median))

    for u,v,img in zip(u_median, v_median, imgs):
        img_pad = np.pad(img, ((u_pad, u_pad), (v_pad, v_pad)), constant_values=np.nan)
        img_pad = np.roll(img_pad, (-u,-v), axis=(0,1))
        imgs_pad.append(img_pad)

    imgs_pad = np.asarray(imgs_pad)
    return imgs_pad
# ==================================================

# --------------------------------------------------
def A5_groundtruth(trace, retrace, std_q=.2):
    assert trace.ndim == 3
    assert trace.shape == retrace.shape

    trace_avg = np.mean(trace, axis=0)
    retrace_avg = np.mean(retrace, axis=0)
    min_avg = np.min([trace_avg, retrace_avg], axis=0)

    trace_std = np.std(trace, axis=0)
    retrace_std = np.std(retrace, axis=0)
    min_std = np.where(trace_avg <= retrace_avg, trace_std, retrace_std)
    std_mask = (min_std <= np.nanquantile(min_std, std_q))
    min_avg[~std_mask] = np.nan

    return min_avg

def calculate_psnr(src, tgt, norm=True, eps=1e-20):
    imgs = src.copy()
    gt = tgt.copy()
    
    if imgs.ndim == 2:
        imgs = np.expand_dims(imgs, axis=0)
    assert imgs.ndim == 3
    assert gt.ndim == 2

    for img in imgs:
        img[np.isnan(gt)] = np.nan

    if norm:
        imgs -= np.nanmin(imgs, axis=(1,2), keepdims=True)
        imgs /= np.nanmax(imgs, axis=(1,2), keepdims=True) + eps
        gt -= np.nanmin(gt)
        gt /= np.nanmax(gt) + eps
    else:
        imgs -= np.nanmedian(imgs, axis=(1,2), keepdims=True)
        gt -= np.nanmedian(gt)

    psnr = []
    for i in range(imgs.shape[0]):
        mse = np.nanmean(np.square(imgs[i] - gt))+eps
        if norm:
            psnr.append(10*np.log10(1/mse))
        else:
            # 12-bit of DAAD board (convert digital signals to heights in nm)
            # height range = 10V * 2 * 11.65 nm/V
            psnr.append(10*np.log10(233**2/mse))

    return np.mean(psnr)

def calculate_ssim(src, tgt, norm=True, c=1e-7, eps=1e-20):
    imgs = src.copy()
    gt = tgt.copy()
    if imgs.ndim == 2:
        imgs = np.expand_dims(imgs, axis=0)
    assert imgs.ndim == 3
    assert gt.ndim == 2

    for img in imgs:
        img[np.isnan(gt)] = np.nan

    if norm:
        imgs -= np.nanmin(imgs, keepdims=True)
        imgs /= np.nanmax(imgs, keepdims=True) + eps
        gt -= np.nanmin(gt)
        gt /= np.nanmax(gt) + eps
    else:
        imgs -= np.nanmedian(imgs, axis=(1,2), keepdims=True)
        gt -= np.nanmedian(gt)

    ssim = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        mask = (~np.isnan(img)) * (~np.isnan(gt))

        if norm:
            ssim.append(structural_similarity(img[mask], gt[mask], data_range=1))
        else:
            # c is determined by a pair of heavily average images
            # height range = 10V * 2 * 11.65 nm/V
            ssim.append(structural_similarity(img[mask], gt[mask], data_range=233, K1=np.sqrt(c), K2=3*np.sqrt(c)))

    return np.mean(ssim)
# ==================================================
