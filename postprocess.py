import os
import numpy as np
from skimage import io
import argparse
from utils import pixelization

class output_generator():
    def __init__(self, checkpoint_idx, prediction_path, data_path):
        self.checkpoint_idx = checkpoint_idx
        self.prediction_path = prediction_path
        self.data_path = data_path
        self.ensembles = [d for d in os.listdir(self.prediction_path) if d.startswith('ensemble')]

        self.x_coord = np.genfromtxt(os.path.join(self.data_path, 'x_coord.csv'), delimiter=',')
        self.t2r_list = [io.imread(os.path.join(self.prediction_path, f'{e}', f't2r_{self.checkpoint_idx}.tif')) for e in self.ensembles]
        self.r2t_list = [io.imread(os.path.join(self.prediction_path, f'{e}', f'r2t_{self.checkpoint_idx}.tif')) for e in self.ensembles]
        self.t2r_list = np.squeeze(self.t2r_list)
        self.r2t_list = np.squeeze(self.r2t_list)

        self.trace = io.imread(os.path.join(self.data_path, 'trace.tif'))
        self.retrace = io.imread(os.path.join(self.data_path, 'retrace.tif'))

    def fuse(self, mov1, mov2):
        assert mov1.shape == mov2.shape
        b,h,w = mov1.shape

        fused_mov = mov2.copy()
        for i in range(b):
            if i:
                fused_mov[i,:int(h*i/b),:] = mov1[i,:int(h*i/b)]
            else:
                fused_mov[i] = mov1[i]
        return fused_mov

    def generate(self):

        output_ensemble = []
        output_ensemble_pix = []
        for i,e in enumerate(self.ensembles):
            if len(self.ensembles) > 1: # multiple ensembles
                output = np.min([self.t2r_list[i], self.r2t_list[i]], axis=0)
            else: # single ensemble
                output = np.min([self.t2r_list, self.r2t_list], axis=0)

            output_pix = pixelization(self.x_coord, output)
            io.imsave(os.path.join(self.prediction_path, f'{e}', f'{e}_output_{self.checkpoint_idx}.tif'), output.astype(np.float32), check_contrast=False)
            io.imsave(os.path.join(self.prediction_path, f'{e}', f'{e}_output_{self.checkpoint_idx}_pix.tif'), output_pix.astype(np.float32), check_contrast=False)

            output_ensemble.append(output)
            output_ensemble_pix.append(output_pix)

        output_ensemble = np.mean(output_ensemble, axis=0)
        output_ensemble_pix = np.mean(output_ensemble_pix, axis=0)
        io.imsave(os.path.join(self.prediction_path, f'output_{self.checkpoint_idx}.tif'), output_ensemble.astype(np.float32), check_contrast=False)
        io.imsave(os.path.join(self.prediction_path, f'output_{self.checkpoint_idx}_pix.tif'), output_ensemble_pix.astype(np.float32), check_contrast=False)

        self.trace_pixelized = pixelization(self.x_coord, self.trace)
        self.retrace_pixelized = pixelization(self.x_coord, self.retrace)
        io.imsave(os.path.join(self.prediction_path, 'trace_pix.tif'), self.trace_pixelized.astype(np.float32), check_contrast=False)
        io.imsave(os.path.join(self.prediction_path, 'retrace_pix.tif'), self.retrace_pixelized.astype(np.float32), check_contrast=False)

        self.fused = self.fuse(output_ensemble_pix, self.trace_pixelized)
        io.imsave(os.path.join(self.prediction_path, f'fused_{self.checkpoint_idx}_pix.tif'), self.fused.astype(np.float32), check_contrast=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--checkpoint_idx', type=str, default='last')
    parser.add_argument('-pp', '--prediction_path', type=str, default='predictions')
    parser.add_argument('-dp', '--data_path', type=str, default='datasets')
    opt = parser.parse_args()

    output_gen = output_generator(
            checkpoint_idx = opt.checkpoint_idx,
            prediction_path = opt.prediction_path,
            data_path = opt.data_path,
            )
    output_gen.generate()