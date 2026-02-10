import numpy as np
from skimage import io
import os
import argparse
from matplotlib import pyplot as plt
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set2.colors)
import matplotlib
matplotlib.rcParams['pdf.fonttype']=42
from einops import rearrange

class data_generator():
    def __init__(self, raw_path, data_path, shift=None, window=.1, tl=.05, tr=.95, rl=0., rr=1., tem=.01, save=False, coef=None):
        self.raw_path = raw_path
        self.file_path = [f for f in os.listdir(self.raw_path) if f.endswith('tif')]
        for f in self.file_path:
            if '1ChRet' in f:
                self.retrace = io.imread(os.path.join(self.raw_path, f))
            else:
                self.trace = io.imread(os.path.join(self.raw_path, f))
        self.data_path = data_path
        self.shift = shift # shift ratio
        self.window = window # window ratio
        self.tl = tl # trace left boundary ratio
        self.tr = tr # trace right boundary ratio
        self.rl = rl # retrace left boundary ratio
        self.rr = rr # retrace right boundary ratio
        self.tem = tem # temperature in masked attn_matrix
        self.save = save
        self.coef = coef

        assert self.trace.shape == self.retrace.shape
        if self.trace.ndim == 2:
            self.trace = rearrange(self.trace, 'h w -> 1 h w')
            self.retrace = rearrange(self.retrace, 'h w -> 1 h w')
        self.b,self.h,self.w = self.trace.shape

    def corr_imgs_on_x(self, img1, img2):
        fimg1 = np.fft.fftshift(np.fft.fft2(img1))
        fimg2 = np.fft.fftshift(np.fft.fft2(img2))
        fcor = fimg1 * np.conj(fimg2)
        fcor = np.sum(fcor, axis=0) # correlation on x axis
        cor = np.fft.ifft(np.fft.fftshift(fcor))
        cor = np.abs(cor)
        return cor

    def global_translation_on_x(self):
        shift = []
        for i1, i2 in zip(self.trace, self.retrace):
            cor = self.corr_imgs_on_x(i1, i2)
            sf = np.argmax(cor)
            shift.append(sf)
        shift = np.mean(shift)

        if shift > self.w//2:
            shift = self.w - shift

        return int(shift)

    def similiarty_matrix(self):
        query = self.trace.copy()
        key = self.retrace.copy()
        assert query.shape == key.shape

        # linear projection - normalized vectors
        query -= np.mean(query, axis=-2, keepdims=True)
        key -= np.mean(key, axis=-2, keepdims=True)
        query = query / np.linalg.norm(query, axis=-2, keepdims=True)
        key = key / np.linalg.norm(key, axis=-2, keepdims=True)

        mat = np.einsum('bki, bkj -> bij', key, query) # transpose(k) @ q
        mat = np.mean(mat, axis=0)
        return mat

    def softmax(self, x, eps):
        x_max = np.max(x, axis=-2, keepdims=True)
        x_max[x_max==-np.inf] = 0 # prevent nan
        z = x - x_max
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=-2, keepdims=True)
        softmax = numerator/(denominator+eps)
        return softmax

    def attn_matrix(self, sim_mat, eps=1e-20):
        sim_scores = sim_mat.copy()
        left_idx = int(self.w*self.tl)
        right_idx = int(self.w*self.tr)
        top_idx = int(self.w*self.rl)
        bottom_idx = int(self.w*self.rr)

        mask = np.ones_like(sim_scores)
        if self.shift is not None:
            mask = np.triu(mask,k=int(self.shift*self.w+self.window*self.w//2)) + np.tril(mask,k=int(self.shift*self.w-self.window*self.w//2)-1)
        else:
            self.shift = self.global_translation_on_x()
            mask = np.triu(mask,k=int(self.shift+self.window*self.w//2)) + np.tril(mask,k=int(self.shift-self.window*self.w//2)-1)
        mask[:,:left_idx] = 1
        mask[:,right_idx:] = 1
        mask[:top_idx,:] = 1
        mask[bottom_idx:,:] = 1

        sim_scores /= self.tem
        attn = self.softmax(sim_scores, eps)
        sim_scores += np.where(mask==1, -np.inf, 0)
        attn_masked = self.softmax(sim_scores, eps)
        return attn, attn_masked, mask.astype(bool)

    def displacement_fit(self, attn_masked, mask):
        yy, xx = np.mgrid[:self.w, :self.w]
        xx = xx[~mask]
        yy = yy[~mask]
        ww = attn_masked[~mask] # weights
        if self.coef is None:
            coef = np.polyfit(xx, yy-xx, 2, w=ww)
        else:
            coef = self.coef

        print(f"quadratic coefficient (*1e-3): \t {np.around(coef[0]*1e3, 4)}")
        print(f"anchored column: \t\t {np.around(-coef[1]/2/(coef[0]+1e-20), 2)} ({np.around(-coef[1]/2/(coef[0]+1e-20)/self.w*100, 1)}%)")
        print(f"translation: \t\t\t {np.around(coef[2]-(coef[1]**2)/4/(coef[0]+1e-20), 2)}")
        print("\nquadratic fitting parameters:")
        print(coef[0])
        print(coef[1])
        print(coef[2])

        return coef, xx, yy, ww

    def boundary_mask(self, coef, mask):
        x_trace_mask = np.sum(~mask, axis=-2).astype(bool)
        x_trace_idx = np.arange(self.w)[x_trace_mask]
        displace = np.poly1d(coef)(x_trace_idx)

        x_retrace_min = np.min(displace+x_trace_idx).astype(int)
        x_retrace_max = np.max(displace+x_trace_idx).astype(int)
        x_retrace_idx = np.arange(self.w)
        x_retrace_mask = np.logical_and((x_retrace_idx >= x_retrace_min), (x_retrace_idx <= x_retrace_max))

        return x_trace_mask, x_retrace_mask

    def displacement_compensate(self, coef, x_trace_mask, x_retrace_mask):
        x_trace = np.arange(self.w, dtype=np.float32)
        x_retrace = np.arange(self.w, dtype=np.float32)
        displace = np.poly1d(coef)(x_trace)

        # align retrace to trace at the minimal point of the displace curve
        delta_ind = int(-coef[1]/(coef[0]+1e-20)/2)
        delta_x = displace[delta_ind]
        x_retrace -= delta_x
        displace -= delta_x
        x_trace[:delta_ind] = x_trace[:delta_ind] + displace[:delta_ind]
        x_retrace[delta_ind:] = x_retrace[delta_ind:] - displace[delta_ind:]

        x_trace = x_trace[x_trace_mask]
        x_retrace = x_retrace[x_retrace_mask]

        return x_trace, x_retrace

    def cross_interp(self, x1, x2, imgs1, imgs2):
        assert imgs1.shape[0] == self.b
        assert imgs2.shape[0] == self.b
        assert imgs1.shape[1] == self.h
        assert imgs2.shape[1] == self.h

        x_min = np.max((np.min(x1), np.min(x2)))
        x_max = np.min((np.max(x1), np.max(x2)))
        x1_mask = np.logical_and((x1 >= x_min), (x1 < x_max))
        x2_mask = np.logical_and((x2 >= x_min), (x2 < x_max))

        x1 = x1[x1_mask]
        x2 = x2[x2_mask]
        imgs1 = imgs1[...,x1_mask]
        imgs2 = imgs2[...,x2_mask]

        x = np.unique(np.concatenate((x1, x2)))
        outs1 = np.zeros((self.b, self.h, len(x)))
        outs2 = np.zeros((self.b, self.h, len(x)))

        for i in range(self.b):
            for j in range(self.h):
                outs1[i,j,:] = np.interp(x, x1, imgs1[i,j,:])
                outs2[i,j,:] = np.interp(x, x2, imgs2[i,j,:])

        return x, outs1, outs2

    def median_line_img(self, img, radius=0):
        assert img.ndim == 2
        assert radius >= 0
        h,w = img.shape
        img_median = np.zeros((h,w))
        img_pad = np.pad(img, ((radius,radius),(0,0)), mode='edge')
        for i in range(h):
            img_median[i,:] = np.median(img_pad[i:i+1+2*radius,:])
        img_median_line = img - img_median
        return img_median_line

    def median_line_subtract(self, imgs):
        if imgs.ndim == 2:
            imgs = np.expand_dims(imgs, axis=0)
        assert imgs.ndim == 3

        imgs_median_line = np.empty_like(imgs)
        for i in range(imgs.shape[0]):
            imgs_median_line[i] = self.median_line_img(imgs[i])
        return imgs_median_line

    def plane_fit_img(self, img):
        assert img.ndim == 2
        h, w = img.shape
        hh, ww = np.mgrid[:h, :w]
        b = img.flatten()
        A = np.column_stack((ww.flatten(), hh.flatten(), np.ones(len(b))))
        x,_,_,_ = np.linalg.lstsq(A,b,rcond=None)
        Cw,Ch,C = x
        z = Cw * ww + Ch * hh + C
        return img - z

    def plane_flatten(self, imgs):
        if imgs.ndim == 2:
            imgs = np.expand_dims(imgs, axis=0)
        assert imgs.ndim == 3

        imgs_flatten = np.empty_like(imgs)
        for i in range(imgs.shape[0]):
            imgs_flatten[i] = self.plane_fit_img(imgs[i])
        return imgs_flatten

    def generate(self):
        self.trace = self.median_line_subtract(self.trace)
        self.trace = self.plane_flatten(self.trace)
        self.retrace = self.median_line_subtract(self.retrace)
        self.retrace = self.plane_flatten(self.retrace)

        sim_mat = self.similiarty_matrix()
        attn, attn_masked, mask = self.attn_matrix(sim_mat)
        coef, xx, yy, ww = self.displacement_fit(attn_masked, mask)
        xtm, xrm = self.boundary_mask(coef, mask) # x_trace_mask, x_retrace_mask

        x_eval = np.arange(self.w)[xtm]
        y_eval = np.poly1d(coef)(x_eval)

        x1, x2 = self.displacement_compensate(coef, xtm, xrm)
        imgs1 = self.trace[...,xtm].copy()
        imgs2 = self.retrace[...,xrm].copy()

        x, imgs1, imgs2 = self.cross_interp(x1, x2, imgs1, imgs2)
        imgs1 = self.plane_flatten(imgs1)
        imgs2 = self.plane_flatten(imgs2)

        # plot
        plt.figure()
        plt.scatter(xx, yy-xx, c=ww, alpha=ww, marker=',')
        plt.plot(x_eval, y_eval)
        plt.xlabel('trace column index')
        plt.ylabel('displace pixels')

        plt.figure()
        plt.imshow(attn_masked, interpolation=None, extent=[0,1,1,0])
        plt.imshow(1-mask, "grey", interpolation=None, alpha=.1, extent=[0,1,1,0])
        plt.plot(x_eval/self.w, (x_eval+y_eval)/self.w, color='white', linestyle='dashed', alpha=.8)
        plt.title('sliding window attention')
        plt.xlabel('trace column index')
        plt.ylabel('retrace column index')

        plt.figure()
        plt.imshow(attn, interpolation=None, extent=[0,1,1,0])
        plt.imshow(1-mask, "grey", interpolation=None, alpha=.1, extent=[0,1,1,0])
        plt.plot(x_eval/self.w, (x_eval+y_eval)/self.w, color='white', linestyle='dashed', alpha=.8)
        plt.title('global attention')
        plt.xlabel('trace column index')
        plt.ylabel('retrace column index')

        plt.show()

        if self.save:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            np.savetxt(os.path.join(self.data_path, 'x_coord.csv'), x, delimiter=',')
            np.savetxt(os.path.join(self.data_path, 'fitting_parameters.csv'), coef, delimiter=',')
            io.imsave(os.path.join(self.data_path, 'trace.tif'), imgs1.astype(np.float32), check_contrast=False)
            io.imsave(os.path.join(self.data_path, 'retrace.tif'), imgs2.astype(np.float32), check_contrast=False)

            io.imsave(os.path.join(self.data_path, 'overlap.tif'), np.concatenate(([np.mean(imgs1, axis=0)], [np.mean(imgs2, axis=0)]), axis=0).astype(np.float32), check_contrast=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'subpixel alignment of AFM columns between trace and retrace')
    parser.add_argument('-rp', '--raw_path', type=str, default='raw', help='raw images path')
    parser.add_argument('-dp', '--data_path', type=str, default='datasets', help='dataset path for training')
    parser.add_argument('-sf', '--shift', type=float, default=None, help='attention area shift')
    parser.add_argument('-w', '--window', type=float, default=.1, help='attention window size')
    parser.add_argument('-tl', '--trace_left', type=float, default=.05, help='trace left boundary ratio')
    parser.add_argument('-tr', '--trace_right', type=float, default=.95, help='trace right boundary ratio')
    parser.add_argument('-rl', '--retrace_left', type=float, default=0., help='retrace left boundary ratio')
    parser.add_argument('-rr', '--retrace_right', type=float, default=1., help='retrace right boundary ratio')
    parser.add_argument('-t', '--temperature', type=float, default=.01, help='temperature in attention')
    parser.add_argument('-s', '--save', action='store_true', help='save resluts')
    parser.add_argument('-c', '--coef', nargs='+', type=float, help='predefined coefficients')
    opt = parser.parse_args()

    data_gen = data_generator(
            raw_path = opt.raw_path,
            data_path = opt.data_path,
            shift = opt.shift,
            window = opt.window,
            tl = opt.trace_left,
            tr = opt.trace_right,
            rl = opt.retrace_left,
            rr = opt.retrace_right,
            tem = opt.temperature,
            save = opt.save,
            coef = opt.coef,
            )
    data_gen.generate()
