# loss function
import torch
import torch.nn.functional as F


def grad_optr(image):

    dx = image.squeeze() - F.pad(image.squeeze(), (0, 1, 0, 0))[:, 1:]
    dy = image.squeeze() - F.pad(image.squeeze(), (0, 0, 0, 1))[1:, :]

    dx[:, -1] = 0
    dy[-1, :] = 0

    return dx, dy

def wrap(vector):
    return torch.remainder(vector + torch.pi, 2*torch.pi)

def loss_fn(predict, target, avg=True):
    assert predict.size() == target.size(), (predict.size(), target.size())
        
    del_pred_x, del_pred_y = grad_optr(predict)
    del_tar_x, del_tar_y = grad_optr(target)

    dx_dy_pred = torch.concat([wrap(del_pred_x).ravel().unsqueeze(dim=1), wrap(del_pred_y).ravel().unsqueeze(dim=1)], dim=1)
    dx_dy_tar_wrap = torch.concat([wrap(del_tar_x).ravel().unsqueeze(dim=1), wrap(del_tar_y).ravel().unsqueeze(dim=1)], dim=1)

    # dx_dy_pred = torch.concat([del_pred_x.ravel().unsqueeze(dim=1), del_pred_y.ravel().unsqueeze(dim=1)], dim=1)
    # dx_dy_tar_wrap = torch.concat([del_tar_x.ravel().unsqueeze(dim=1), del_tar_y.ravel().unsqueeze(dim=1)], dim=1)
    
    norm = torch.linalg.norm(dx_dy_pred- dx_dy_tar_wrap, dim=1)
    loss = torch.sum(norm)

    if avg:
        return loss/norm.size()[0]
    else:
        return loss

def total_variation_loss(img, weight, l=1, th=0.2):
    h_img, w_img = img.size()
    
    if l == 1:
        tv_h = torch.abs(img[1:,:]-img[:-1,:]).sum()
        tv_w = torch.abs(img[:,1:]-img[:,:-1]).sum()
    elif l == 2:
        tv_h = torch.pow(img[1:,:]-img[:-1,:], 2).sum()
        tv_w = torch.pow(img[:,1:]-img[:,:-1], 2).sum()
    elif l == 0:
        tv_h = (torch.abs(img[1:,:]-img[:-1,:]) > th).sum()
        tv_w = (torch.abs(img[:,1:]-img[:,:-1]) > th).sum()
    else:
        return 0


    return weight*(tv_h+tv_w)/(h_img*w_img)

def l2_loss(img, target):
    return torch.norm((img - target).ravel())/target.ravel().size()[0]