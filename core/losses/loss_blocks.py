import torch
import torch.nn as nn
import torch.nn.functional as F

def robust_l1(x):
  """Robust L1 metric."""
  return (x**2 + 0.001**2)**0.5

# # Credit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py

def abs_robust_loss(diff, eps=0.01, q=0.4):
    """The so-called robust loss used by DDFlow."""
    return torch.pow((torch.abs(diff) + eps), q)


def TernaryLoss(im, im_warp, occ, patch_size=7):
    
    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv2d(intensities, weights, padding=(patch_size//2))
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist) #b, 49, h, w
        dist_sum = torch.sum(dist_norm, 1, keepdim=True) 

        return dist_sum

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask*t 

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = abs_robust_loss(_hamming_distance(t1, t2)) #apply robust loss...
    mask = _valid_mask(occ, (patch_size//2)) # invalid borders and occ combined.
    masked_sum = dist * mask
    masked_sum = torch.sum(masked_sum) / (torch.sum(mask.detach())+ 0.000001) # mean over the whole batch - masked. SMURF --> but we where already doing this out of this function.
   
    return masked_sum
    
    
def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist


def gradient(data, stride=1): #  TO REPRODUCE SMURF.
    D_dy = data[:, :, stride:] - data[:, :, :-stride]
    D_dx = data[:, :, :, stride:] - data[:, :, :, :-stride]
    return D_dx, D_dy


def smooth_grad_1st(flo, image, alpha=10):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * robust_l1(dx) / 2.
    loss_y = weights_y * robust_l1(dy) / 2

    return loss_x.mean() / 2. + loss_y.mean() / 2.


def smooth_grad_2nd(flo, image, alpha=10):
    img_dx, img_dy = gradient(image, stride=2) 
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
    
    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x * robust_l1(dx2)
    loss_y = weights_y * robust_l1(dy2)

    return loss_x.mean() / 2. + loss_y.mean() / 2.
