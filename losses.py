import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn.functional import binary_cross_entropy_with_logits

import pytorch_msssim
from skimage import draw

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'

def compute_discriminator_loss(fake_logits, real_logits):
    fake_gt = torch.zeros(fake_logits.shape)
    real_gt = torch.ones(real_logits.shape)
    if is_cuda:
        fake_gt = fake_gt.cuda()
        real_gt = real_gt.cuda()

    d_fake_loss = binary_cross_entropy_with_logits(fake_logits, fake_gt)
    d_real_loss = binary_cross_entropy_with_logits(real_logits, real_gt)

    d_loss = 0.5 * (d_real_loss + d_fake_loss)

    return d_loss

def compute_r1_loss(real_latent, real_logits):
    r1_gamma = 10
    # Reference >> https://github.com/rosinality/style-based-gan-pytorch/blob/a3d000e707b70d1a5fc277912dc9d7432d6e6069/train.py
    # little different with original DiracGAN
    grad_real = grad(outputs=real_logits.sum(), inputs=real_latent, create_graph=True)[0]
    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
    grad_penalty = 0.5*r1_gamma*grad_penalty
    return grad_penalty

def compute_generator_loss(fake_logits):
    fake_gt = torch.ones(fake_logits.shape)
    if is_cuda:
        fake_gt = fake_gt.cuda()

    d_fake_loss = binary_cross_entropy_with_logits(fake_logits, fake_gt)

    d_loss = 1. * d_fake_loss

    return d_loss

def compute_id_loss(src_emb, generated_emb):
    return torch.mean(F.l1_loss(generated_emb, src_emb))
    
def compute_latent_loss(orig_latent, fake_latent):
    return torch.mean(F.l1_loss(orig_latent, fake_latent))

def compute_emotion_loss(src_emotions, generated_emotions):
    return F.mse_loss(generated_emotions, src_emotions, reduction="mean")

def compute_landmark_loss(src_landmarks, src_landmarks_maxval, generated_landmarks, generated_landmarks_maxval, landmark_mask):
    mask = (src_landmarks_maxval > 0.6) * (generated_landmarks_maxval > 0.6)
    mask = mask.squeeze_(2) * landmark_mask
    if mask.sum() == 0:
        landmark_loss = torch.Tensor([0.]).float().to('cuda' if is_cuda else 'cpu')
    else:
        landmark_loss = F.mse_loss(generated_landmarks[mask], src_landmarks[mask], reduction="mean")
    return landmark_loss

def compute_pixel_reconstruction_loss(gt_images, generated_images, with_mssim=True):
    l1_loss = F.l1_loss(gt_images, generated_images, reduction="mean") # sample_weight=self.pixel_mask
    # l1_loss = F.l1_loss(gt_images, generated_images, reduction="sum") # sample_weight=self.pixel_mask
    # if args.pixel_loss_type == 'mix':
    if with_mssim:
        mssim = torch.mean(1 - pytorch_msssim.ms_ssim(gt_images, generated_images, data_range=1.))
        pixel_loss = (0.84 * mssim + 0.16 * l1_loss)
    else:
        pixel_loss = l1_loss

    return pixel_loss

def compute_l2_loss(gt_images, generated_images):
    return F.mse_loss(gt_images, generated_images, reduction="mean")     

def compute_background_mask(images, src_landmarks, src_landmarks_maxval):
    valid_masks = (src_landmarks_maxval > 0.6).squeeze_(2)
    masks = []
    valid_bg = True
    for image, src_lnd, valid_mask in zip(images, src_landmarks, valid_masks):
        src_lnd = src_lnd[list(range(0, 17)) + list(range(26, 20, -1)) + list(range(21, 16, -1))]
        valid_mask = valid_mask[list(range(0, 17)) + list(range(26, 20, -1)) + list(range(21, 16, -1))]
        if valid_mask.sum() < 5:
            mask = np.zeros(image.shape[1:]).astype(np.bool8)
            valid_bg = False
        else:
            mask = draw.polygon2mask(
                image.shape[1:],
                src_lnd[valid_mask].cpu().numpy()[:,::-1]
            )
            mask = mask == 0
        mask = np.expand_dims(mask, axis=0).repeat(3, 0)
        masks.append(mask)
    return np.array(masks), valid_bg

def compute_background_loss(background_masks, gt_images, generated_images):
    return compute_pixel_reconstruction_loss(
        gt_images[torch.from_numpy(background_masks)],
        generated_images[torch.from_numpy(background_masks)],
        with_mssim=False
    )
    
__all__ = [
    'compute_id_loss',
    'compute_emotion_loss',
    'compute_latent_loss',
    'compute_pixel_reconstruction_loss',
    'compute_l2_loss',
    'compute_background_mask',
    'compute_background_loss',
    'compute_discriminator_loss',
    'compute_r1_loss',
    'compute_generator_loss',
    'compute_landmark_loss'
]
