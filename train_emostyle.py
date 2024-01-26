import os
import pickle
import random
import argparse


import matplotlib.pyplot as plt
import numpy as np
import torch
from typing_extensions import final
from PIL import Image
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable, grad
from torch.nn.functional import binary_cross_entropy_with_logits
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm

import pytorch_msssim
from dataset import SyntheticDataset
from models.emo_mapping import EmoMappingW, EmoMappingWplus
from models.landmark import FaceAlignment
from models.emonet import EmoNet
from models.vggface2 import VGGFace2
from models.stylegan2_interface import StyleGAN2
from losses import *
from skimage import draw
import glob

is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'
EMO_EMBED = 64
STG_EMBED = 512
INPUT_SIZE = 1024

def train(
    datapath: str,
    stylegan2_checkpoint_path: str,
    vggface2_checkpoint_path: str,
    emonet_checkpoint_path: str,
    log_path: str,
    output_path: str,
    wplus: bool
):
    stylegan = StyleGAN2(
        checkpoint_path=stylegan2_checkpoint_path,
        stylegan_size=INPUT_SIZE,
        is_dnn=True,
        is_pkl=True
    )
    stylegan.eval().requires_grad_(False)

    vggface2_net = VGGFace2(vggface2_checkpoint_path)
    vggface2_net.eval()

    landmark_net = FaceAlignment()
    landmark_net.eval()

    face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
    
    ckpt_emo = torch.load(emonet_checkpoint_path)
    ckpt_emo = { k.replace('module.',''): v for k,v in ckpt_emo.items() }
    emonet = EmoNet(n_expression=8)
    emonet.load_state_dict(ckpt_emo, strict=False)
    emonet.eval()

    
    if wplus:
        emo_mapping = EmoMappingWplus(INPUT_SIZE, EMO_EMBED, STG_EMBED)
    else:
        emo_mapping = EmoMappingW(EMO_EMBED, STG_EMBED)

    
    kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(SyntheticDataset(datapath, mode='training'), batch_size=4, shuffle=True, **kwargs)

    lr = 5e-5
    generator_params = list(emo_mapping.parameters())
    generator_optimizer = optim.Adam(generator_params, lr=lr, betas=(0.9, 0.999), eps=1e-08) # weight_decay=1e-4
    generator_gan_optimizer = optim.Adam(generator_params, lr=lr * 0.1, betas=(0.9, 0.999), eps=1e-08)

    
    if is_cuda:
        stylegan.cuda()
        vggface2_net.cuda()
        landmark_net.cuda()
        emonet.cuda()
        face_pool.cuda()
        emo_mapping.cuda()

    weight_id = 1.5
    weight_emotion = 0.3
    weight_background = 0.2
    weight_landmark = 0.001
    weight_reconstruction = 0.2
    weight_latent = 1
    weight_latent_regularizer = 0.03


    landmark_masks = {
        'same_face': torch.BoolTensor(torch.ones([68]).bool()), # same face
        'emo_face': torch.BoolTensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, # lower face [0:16]
                            0, 0, 0, 0, 0, # left eyebrow [17:21]
                            0, 0, 0, 0, 0, # right eyebrow [22:26]
                            1, 1, 1, 1, 0, 0, 0, 0, 0, # nose
                            0, 0, 0, 0, 0, 0, # left eye
                            0, 0, 0, 0, 0, 0, # right eye
                            0, 0, 0, 0, 0, 0, # upper lip
                            0, 0, 0, 0, 0, 0, # lower lip
                            0, 0, 0, 0, 0, 0, 0, 0]),
    }
    if is_cuda:
        landmark_masks['same_face'] = landmark_masks['same_face'].cuda()
        landmark_masks['emo_face'] = landmark_masks['emo_face'].cuda()

    iter = 1
    for epoch in range(6):
        emo_mapping.train()
        for images, latents in train_loader:
            if wplus:
                latents = latents
            else:
                latents = latents[:, 0, :]
            
            if is_cuda:
                images, latents = images.cuda(), latents.cuda()
            images = Variable(images)
            latents = Variable(latents)
            latents.requires_grad = True

            batch_size = images.shape[0]

            same_face = iter % 10 < 2
            
            loss_log = {}

            emo_mapping.zero_grad()
            with torch.no_grad():
                src_ids = vggface2_net(images)
                orig_latents = latents
                src_landmarks, src_landmarks_maxval = landmark_net(images)
                background_masks, valid_bg_mask = compute_background_mask(images, src_landmarks, src_landmarks_maxval)
                src_emotions = emonet(images)
                random_emotion = torch.FloatTensor(batch_size, 2).uniform_(-1., 1.).cuda()
                if same_face:
                    current_emotion = src_emotions
                else:
                    current_emotion = random_emotion

                    
            latent_diff = emo_mapping(orig_latents, current_emotion)
            fake_latents = orig_latents + latent_diff

            generated_images = stylegan.generate(fake_latents)
            generated_images = (generated_images + 1.) / 2.
            generated_images = face_pool(generated_images)
            generated_images_ids = vggface2_net(generated_images)
            
            id_loss = weight_id * \
                compute_id_loss(src_ids, generated_images_ids)
            loss_log['id_loss'] = "{:.4f}".format(id_loss.item())
            
            generated_emotions = emonet(generated_images)
            emotion_loss = weight_emotion * \
                compute_emotion_loss(current_emotion, generated_emotions)
            loss_log['emo_loss'] = "{:.4f}".format(emotion_loss.item())

            generated_landmarks, generated_landmarks_maxval = landmark_net(generated_images)
            landmark_mask = landmark_masks['same_face' if same_face else 'emo_face'].repeat(batch_size).view(batch_size, -1)
            landmark_loss = weight_landmark * \
                    compute_landmark_loss(src_landmarks, src_landmarks_maxval, generated_landmarks, generated_landmarks_maxval, landmark_mask)
            loss_log['landmarks'] = "{:.4f}".format(landmark_loss.item())

            if same_face:
                latent_loss = weight_latent * \
                    compute_latent_loss(orig_latents, fake_latents)
                loss_log['latent_loss'] = "{:.4f}".format(latent_loss.item())
                reconstruction_loss = weight_reconstruction * \
                    compute_pixel_reconstruction_loss(images, generated_images)
                loss_log['recon_loss'] = "{:.4f}".format(reconstruction_loss.item())
                background_loss = 0
            else:
                if valid_bg_mask:
                    background_loss = weight_background * \
                        compute_background_loss(background_masks, images, generated_images)
                    loss_log['bg_loss'] = "{:.4f}".format(background_loss.item())
                else:
                    background_loss = 0
                latent_loss = 0.
                reconstruction_loss = 0.

            # Latent Regularizer
            latent_diff_regularizer = weight_latent_regularizer * \
                torch.norm(latent_diff, p=2, dim=-1).mean()
            loss_log['latent_reg'] = "{:.4f}".format(latent_diff_regularizer.item())

            total_non_gan_losses = id_loss + \
                emotion_loss + \
                reconstruction_loss + \
                background_loss + \
                latent_loss + \
                landmark_loss + \
                latent_diff_regularizer

            loss_log['sum_non_gan'] = "{:.4f}".format(total_non_gan_losses.item())

            generator_optimizer.zero_grad()
            total_non_gan_losses.backward()
            generator_optimizer.step()

            print(epoch, loss_log)

            if iter % 10 == 8:
                if same_face:
                    filename = '{:06}_same_face.png'.format(iter)
                else:
                    filename = '{:06}_{:.2f}_{:.2f}_emo_face.png'.format(iter, src_emotions[0, 0].item(), src_emotions[0, 1].item())
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                if not os.path.exists(os.path.join(log_path, "emo_mapping_training")):
                    os.makedirs(os.path.join(log_path, "emo_mapping_training"))
                
                save_image([images[0], generated_images[0]],
                            os.path.join(log_path, "emo_mapping_training", filename), 
                            normalize=True)

            iter += 1

        torch.save({
                'epoch': epoch,
                'emo_mapping_state_dict': emo_mapping.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'generator_gan_optimizer_state_dict': generator_gan_optimizer.state_dict(),
            }, os.path.join(output_path, 'checkpoint_{}.pt'.format(epoch)))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train script for your program")
    
    parser.add_argument("--datapath", required=True, help="Path to the dataset")
    parser.add_argument("--stylegan2_checkpoint_path", required=True, help="Path to the StyleGAN2 checkpoint")
    parser.add_argument("--vggface2_checkpoint_path", required=True, help="Path to the VGGFace2 checkpoint")
    parser.add_argument("--emonet_checkpoint_path", required=True, help="Path to the Emonet checkpoint")
    parser.add_argument("--log_path", required=True, help="Path to the log directory")
    parser.add_argument("--output_path", required=True, help="Path to the output directory")
    parser.add_argument("--wplus", action="store_true", help="Enable wplus (if provided)")

    args = parser.parse_args()

    train(
        datapath=args.datapath,
        stylegan2_checkpoint_path=args.stylegan2_checkpoint_path,
        vggface2_checkpoint_path=args.vggface2_checkpoint_path,
        emonet_checkpoint_path=args.emonet_checkpoint_path,
        log_path=args.log_path,
        output_path=args.output_path,
        wplus=args.wplus
    )