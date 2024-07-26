import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from wave_prop_module import Wave2d
from torch import optim
from loss_fns import *
from unet_v2 import *
import json


EPOCHS = 11
INP_CHANNELS = 16


# complete model
device = torch.device('cuda')
print(device)

model_amp = UNet2(num_input_channels=INP_CHANNELS, num_layers_to_concat = 8, upsample_mode='bilinear', 
             pad='zero', norm_layer=nn.BatchNorm2d, need_sigmoid=True, need_bias=True).to(device)
model_phase = UNet2(num_input_channels=INP_CHANNELS, num_layers_to_concat = 8, upsample_mode='bilinear', 
             pad='zero', norm_layer=nn.reBatchNorm2d, need_sigmoid=True, need_bias=True).to(device)

# incident wave
X, Y = np.meshgrid(np.linspace(-1, 1, 1024), np.linspace(-1, 1, 1360))

A = torch.tensor([X.T.ravel(), Y.T.ravel(), X.T.ravel()*Y.T.ravel(), X.T.ravel()*X.T.ravel(), Y.T.ravel()*Y.T.ravel(), np.ones_like(X.T.ravel()), 
              (X.T.ravel())**3, (Y.T.ravel())**3, X.T.ravel()*(Y.T.ravel())**2, Y.T.ravel()*(X.T.ravel())**2], dtype=torch.float32).T


# coeffs = torch.rand(10, 1, requires_grad=True)
# coeffs = torch.FloatTensor(10, 1).uniform_(-1, 1)
coeffs_phase = torch.tensor([[ -13.53], [-113.3250], [ -12.7425], [   4.4154], [   6.2158], 
                       [ -119.261], [   2.3329], [  1.3617], [   1.5364], [  3.5796]])
coeffs_phase.requires_grad = True

# obj_amp = torch.rand(1024, 1360, requires_grad=True).to(device)
data = torch.rand(1, INP_CHANNELS, 1024, 1360).float().to(device)

# obj_phase = torch.rand(1024, 1360, requires_grad=True).to(device)

phase_img = np.load('phase_0.npy')
abs_img = np.load('abs_0.npy')

target_phase = torch.tensor(phase_img).T.to(device)
target_abs = torch.tensor(abs_img).T.to(device)

# cons = torch.tensor(3.2, requires_grad=True)
# dist = torch.tensor(155.003, requires_grad=True) # mm 

# loss_fn_alex = lpips.LPIPS(net='alex') 

optimizer = optim.Adam([
    {'params': coeffs_phase, 'lr': 1e-1},
    {'params': model_amp.parameters(), 'lr': 1e-3},
    {'params': model_phase.parameters(), 'lr': 1e-3}],
    lr=0.01, betas=(0.9, 0.999)
    )

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

wave_obj = Wave2d(
        numPx = list(phase_img.shape), 
        sizePx = [6*1e-3, 6*1e-3], 
        wl = 550*1e-6
        )


#========================================================================


loss_arr = []
loss_amp_arr = []
loss_phase_arr = []

for itr in range(EPOCHS):

    optimizer.zero_grad()

    obj_amp = model_amp(data.detach()).squeeze()
    obj_phase = model_phase(data.detach()).squeeze()
    obj = obj_amp*torch.exp(1j*obj_phase)
    
    wave_phase = ((A)@coeffs_phase).reshape(1024, 1360)
    wave = torch.exp(1j*wave_phase).to(device)
    
    wave_op = wave*obj
    wave_obj.wavefield(wave_op.T)
    dist = torch.tensor(155.003) # mm
    wave_at_img = wave_obj.propogate(dist)

    wave_z_amp = torch.abs(wave_at_img).float().to(device)
    wave_z_phase = torch.angle(wave_at_img).float().to(device)

    # loss_proc = 0.9*l2_loss(proc, obj_phase) + total_variation_loss(proc, 0.1)
    
    loss_amp = loss_fn(wave_z_amp, target_abs)
    loss_phase = loss_fn(wave_z_phase, target_phase)
    # loss_amp = l2_loss(torch.abs(wave_at_img).float(), target_abs.float())

    loss = 0.5*loss_amp + 0.5*loss_phase
    # + total_variation_loss(obj_phase, 0.2) + total_variation_loss(obj_amp, 0.1)
    # + 0.1*l2_loss(wave_z_amp, wave_z_amp_proc) + 0.1*l2_loss(wave_z_phase, wave_z_phase_proc)
    # + 1/(grad_x.ravel().abs().mean() + grad_y.ravel().abs().mean())

    # 

    loss.backward()
    
    optimizer.step()
    scheduler.step()

    loss_arr.append(loss.item())
    loss_amp_arr.append(loss_amp.item())
    loss_phase_arr.append(loss_phase.item())

    if (itr+1) % 5 == 0:
        np.save(f'obj_amp_{itr+1}.npy', obj_amp.detach().cpu().numpy())
        np.save(f'obj_phase_{itr+1}.npy', obj_phase.detach().cpu().numpy())
        np.save(f'wave_phase_{itr+1}.npy', wave_phase.detach().cpu().numpy())


save_losses = {
    'loss_total': loss_arr,
    'loss_amp': loss_amp_arr,
    'loss_phase': loss_phase_arr
}

with open('losses.json', 'w') as f:
    json.dump(save_losses, f)

