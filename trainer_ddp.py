import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import json
import math
from models.unet import Unet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import argparse
from utils import imgs_sets, load_data

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class train_net():
    def __init__(self, data_path, mode='t2r', patch_size=(64,64), overlap_th=.2, augmentation=False, qmin=0, qmax=.99, checkpoint_path='checkpoints', bs=512, lr=1e-4, b1=.9, b2=.999, epochs=None, iteration=200, loss_weight=1, ensemble=0):
        self.data_path = data_path
        self.patch_size = patch_size
        self.overlap_th = overlap_th
        self.augmentation = augmentation
        self.qmin = qmin
        self.qmax = qmax
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.ensemble = ensemble
        self.checkpoint_save_path = os.path.join(self.checkpoint_path, f"ensemble{self.ensemble}")
        if not os.path.exists(self.checkpoint_save_path):
            os.makedirs(self.checkpoint_save_path, exist_ok=True)
        self.model = Unet(filter_base=32, unet_depth=3)
        # self.model = SwinIR(img_size=63, window_size=7, upscale=1, in_chans=1, upsampler='',)
        self.bs = bs
        self.epochs = epochs
        self.iteration = iteration
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.loss_weight = loss_weight
        self.MSELoss = torch.nn.MSELoss(reduction='mean')
        self.L1Loss = torch.nn.L1Loss(reduction='mean')

    def train(self, rank, world_size):
        torch.cuda.set_device(rank)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        model = model.to(rank)
        ddp_model = DDP(model, device_ids=[rank])

        data_A, data_B, _, _, _ = load_data(self.data_path, self.mode, rank, self.augmentation, self.qmin, self.qmax, self.patch_size, self.overlap_th)
        dataset = imgs_sets(data_A, data_B)
        sampler = DistributedSampler(dataset, world_size, rank)
        dataloader = DataLoader(dataset, batch_size=self.bs, sampler=sampler)

        if self.epochs is None:
            self.epochs = round(self.iteration / math.ceil(len(dataset)/self.bs/world_size))
            self.epochs = self.epochs if self.epochs else 1

        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr*25, steps_per_epoch=len(dataloader), epochs=self.epochs)

        avg_loss_history = []
        avg_lr_history = []

        iter_count = 0
        with tqdm(range(self.epochs*len(dataloader)), desc=f"ensemble{self.ensemble}-{self.mode}", disable = (rank!=0)) as pbar:
            for epoch in range(self.epochs):
                model.train()
                sampler.set_epoch(epoch)
                for img_A, img_B in dataloader:
                    img_A_pred = ddp_model(img_A)

                    loss = self.loss_weight * self.MSELoss(img_A_pred, img_B) + (1-self.loss_weight) * self.L1Loss(img_A_pred, img_B)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    pbar.update()

                    avg_loss = torch.tensor(0, dtype=torch.float32).to(rank)
                    avg_loss += loss.item()
                    dist.barrier()
                    dist.reduce(avg_loss, dst=0)
                    avg_loss /= dist.get_world_size()
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    if rank == 0:
                        avg_loss_history.append(avg_loss.cpu().numpy())
                        avg_lr_history.append(lr)

                        # if (iter_count+1)%200 == 0:
                           # torch.save(ddp_model.module.state_dict(), f"{self.checkpoint_save_path}/{self.mode}_checkpoint_{iter_count+1}.pth")
                        # iter_count += 1

        if rank == 0:
            torch.save(ddp_model.module.state_dict(), f"{self.checkpoint_save_path}/{self.mode}_checkpoint_last.pth")
            with open(f"{self.checkpoint_save_path}/{self.mode}_loss.csv", 'w') as f:
                for l in avg_loss_history:
                    f.write(str(l)+'\n')
            with open(f"{self.checkpoint_save_path}/{self.mode}_lr.csv", 'w') as f:
                for l in avg_lr_history:
                    f.write(str(l)+'\n')

def train_worker(rank, world_size, opt, mode, ensemble_idx):
    setup(rank, world_size)

    trainer = train_net(
            data_path = opt.data_path,
            patch_size = opt.patch_size,
            overlap_th = opt.overlap_th,
            augmentation = opt.augmentation,
            qmin = opt.qmin,
            qmax = opt.qmax,
            mode = mode,
            checkpoint_path = opt.checkpoint_path,
            bs = opt.batch_size,
            lr = opt.lr,
            b1 = opt.b1,
            b2 = opt.b2,
            epochs = opt.epochs,
            iteration= opt.iteration,
            loss_weight = opt.loss_weight,
            ensemble = ensemble_idx,
            )
    trainer.train(rank, world_size)

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('-e', '--epochs', type=int, default=None)
    parser.add_argument('-it', '--iteration', type=int, default=200)
    parser.add_argument('-ps', '--patch_size', type=int, nargs='+', default=[64])
    parser.add_argument('-th', '--overlap_th', type=float, default=.2)
    parser.add_argument('-w', '--loss_weight', type=float, default=1)
    parser.add_argument('-a', '--augmentation', action='store_true')
    parser.add_argument('--qmin', type=float, default=0)
    parser.add_argument('--qmax', type=float, default=.99)
    parser.add_argument('-dp', '--data_path', type=str, default='datasets')
    parser.add_argument('-cp', '--checkpoint_path', type=str, default='checkpoints')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--b1', type=float, default=.9)
    parser.add_argument('--b2', type=float, default=.999)
    parser.add_argument('-n', '--ensembles', type=int, default=3)
    parser.add_argument('-ws', '--world_size', type=int, default=torch.cuda.device_count())
    opt = parser.parse_args()
    assert opt.world_size <= torch.cuda.device_count()
    if len(opt.patch_size) == 1:
        opt.patch_size = (opt.patch_size[0], opt.patch_size[0])
    elif len(opt.patch_size) == 2:
        opt.patch_size = (opt.patch_size[0], opt.patch_size[1])
    else:
        raise ValueError("patch_size must be 1 or 2 integers")

    print(f"training {opt.ensembles} ensemble models")
    for i in range(opt.ensembles):
        for mode in ['t2r','r2t']:
            mp.spawn(train_worker, args=(opt.world_size, opt, mode, i), nprocs=opt.world_size, join=True)

    with open(f"{opt.checkpoint_path}/hyperparams.txt", 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
