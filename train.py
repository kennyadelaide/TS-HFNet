import os
from torch.utils.data import DataLoader
from weights_init import weights_init
from Loss import HybridLoss
from new_model import UNet3D
from tqdm import tqdm
from brats21_tools.brats import BraTS21Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import torch
import numpy as np
import random


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def set_seed(seed=1000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  #


set_seed(1000)

num_cls = 4
learning_rate = 3e-4
epochs = 150
start_epoch = 0
batch_size = 2
output = './work_dir'


ce_weights = torch.tensor([1.0, 4.0, 4.0, 3.0]).cuda()  # background, ET, TC, WT


loss = HybridLoss(
    ce_weights=ce_weights,
    alpha=0.5,
    beta=0.5,
    gamma=0.5,
    tversky_alpha=0.6,
    tversky_beta=0.4,
    focal_gamma=1.5,
    region_weights=[1.2, 1.2, 1.0]
)

is_pretrained = True


pretrained_model = r'F:\experiment\work_dir\数据集对比实验\TS-HFNet 21\fold5\module_58.pth'

train_data = BraTS21Dataset('F:\\experiment\\dataset\\brats2021\\train', mode='train')
train_load = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=1,
                        pin_memory=True, persistent_workers=True)

val_data = BraTS21Dataset('F:\\experiment\\dataset\\brats2021\\test', mode='test')
val_load = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

if is_pretrained:
    weight_dict = torch.load(pretrained_model, map_location='cpu')
    model = UNet3D(out_channels=num_cls, modal_numbers=4)
    # model = NvNet(config=config)
    model.load_state_dict(weight_dict['model_state_dict'])
    model = model.cuda()
else:
    model = UNet3D(out_channels=num_cls, modal_numbers=4)
    # model = NvNet(config=config)
    weights_init(model, init_type='kaiming', verbose=True)
    model = model.cuda()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-5,
    eps=1e-7,
    betas=(0.9, 0.999)
)


if is_pretrained:
    optimizer.load_state_dict(weight_dict['optimizer_state_dict'])


lr_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=30,
    T_mult=1,
    eta_min=1e-6
)


if __name__ == '__main__':

    scaler = GradScaler(init_scale=8192, growth_interval=epochs, enabled=True)

    for epoch in range(start_epoch, epochs):

        model.train()
        train_loss = 0.
        for batch in tqdm(train_load):
            [modals, targets] = batch
            optimizer.zero_grad()
            modals = modals.to(device='cuda', non_blocking=True)
            targets = targets.to(device='cuda', non_blocking=True).long()

            with autocast():
                scores = model(modals)
                losses = loss(scores, targets)

            train_loss = train_loss + losses.item()

            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        train_loss = train_loss / len(train_load)

        print("train epoch:{}, Loss:{}".format(epoch, train_loss))

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': train_loss,
        }, "{}/module_{}.pth".format(output, epoch + 1))


        val_loss=0.
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_load):
                [modals, targets] = batch
                modals = modals.to(device='cuda', non_blocking=True)
                targets = targets.to(device='cuda', non_blocking=True).long()
                scores = model(modals)
                losses = loss(scores, targets)
                val_loss = val_loss + losses.item()

        lr_schedule.step(val_loss/len(val_load))





