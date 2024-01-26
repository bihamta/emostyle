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
from lpips import LPIPS

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
ITER_NUM = 500

def train(
    datapath: str,
    stylegan2_checkpoint_path: str,
    emo_mapping_checkpoint_path: str,
    vggface2_checkpoint_path: str,
    emonet_checkpoint_path: str,
    log_path: str,
    inversion_type: str,
    output_path: str,
    wplus: bool
):
    stylegan = StyleGAN2(
        checkpoint_path=stylegan2_checkpoint_path,
        stylegan_size=INPUT_SIZE,
        is_dnn=True,
        is_pkl=True
    )
    # Fine Tuning StyleGAN2
    stylegan.train().requires_grad_(True)
    
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

    ckpt_emo_mapping = torch.load(emo_mapping_checkpoint_path)
    if wplus:
        emo_mapping = EmoMappingWplus(INPUT_SIZE, EMO_EMBED, STG_EMBED)    
    else:
        emo_mapping = EmoMappingW(EMO_EMBED, STG_EMBED)
    
    emo_mapping.load_state_dict(ckpt_emo_mapping['emo_mapping_state_dict'])
    emo_mapping.eval()
    
    kwargs = {'num_workers': 1, 'pin_memory': False} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(PersonalizedSyntheticDataset(datapath, inversion_type=inversion_type), batch_size=1, shuffle=False, **kwargs)

    lpips = LPIPS(net='alex').to('cuda' if is_cuda else 'cpu').eval()
    
    optimizer = torch.optim.Adam(stylegan.parameters(), lr=3e-4)

    if is_cuda:
        stylegan.cuda()
        vggface2_net.cuda()
        landmark_net.cuda()
        emonet.cuda()
        face_pool.cuda()
        emo_mapping.cuda()
        
    weight_id = 1.5
    weight_emotion = 0.2
    weight_background = 0.2
    weight_landmark = 0.001
    weight_reconstruction = 0.2
    weight_l2 = 1
    weight_lpips = 1
    
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
    for epoch in range(ITER_NUM):
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

            images_scaled = face_pool(images)

            batch_size = images.shape[0]
            # images = (images + 1.) / 2.
            same_face = iter % 10 < 2
            #same_face = False
            loss_log = {}

            stylegan.zero_grad()
            
            with torch.no_grad():
                src_ids = vggface2_net(images_scaled)
                orig_latents = latents
                src_landmarks, src_landmarks_maxval = landmark_net(images_scaled)
                background_masks, valid_bg_mask = compute_background_mask(images_scaled, src_landmarks, src_landmarks_maxval)

                if same_face:
                    src_emotions = emonet(images)
                else:
                    src_emotions = torch.FloatTensor(batch_size, 2).uniform_(-.7, .7).cuda() # limited the valence and arousal to [-7, 7] instead [-1, 1]

            fake_latents = orig_latents + emo_mapping(orig_latents, src_emotions)

            generated_images = stylegan.generate(fake_latents)
            generated_images = (generated_images + 1.) / 2.
            generated_images_scaled = face_pool(generated_images)
            generated_images_ids = vggface2_net(generated_images_scaled)
            
            id_loss = 0
            id_loss = weight_id * \
                compute_id_loss(src_ids, generated_images_ids)
            loss_log['id_loss'] = "{:.4f}".format(id_loss.item())
            
            generated_emotions = emonet(generated_images)
            emotion_loss = weight_emotion * \
                compute_emotion_loss(src_emotions, generated_emotions)
            loss_log['emo_loss'] = "{:.4f}".format(emotion_loss.item())

            generated_landmarks, generated_landmarks_maxval = landmark_net(generated_images_scaled)
            landmark_mask = landmark_masks['same_face' if same_face else 'emo_face'].repeat(batch_size).view(batch_size, -1)
            landmark_loss = weight_landmark * \
                    compute_landmark_loss(src_landmarks, src_landmarks_maxval, generated_landmarks, generated_landmarks_maxval, landmark_mask)
            loss_log['landmarks'] = "{:.4f}".format(landmark_loss.item())

            if same_face:
                # Sameface loss 1
                # latent_loss = weight_latent * \
                #     compute_latent_loss(orig_latents, fake_latents)
                # loss_log['latent_loss'] = "{:.4f}".format(latent_loss.item())
                # reconstruction_loss = weight_reconstruction * \
                #     compute_pixel_reconstruction_loss(images, generated_images)
                # loss_log['recon_loss'] = "{:.4f}".format(reconstruction_loss.item())
                
                # Sameface loss 2
                l2_loss = weight_l2 * \
                    compute_l2_loss(images_scaled, generated_images)

                lpips_loss = weight_lpips * \
                    torch.squeeze(lpips(generated_images, images_scaled))
                loss_log['lpips_loss'] = "{:.4f}".format(lpips_loss.item())
                
                reconstruction_loss = l2_loss + lpips_loss
                background_loss = 0

            else:
                if valid_bg_mask:
                    background_loss = weight_background * \
                        compute_background_loss(background_masks, images_scaled, generated_images_scaled)
                    loss_log['bg_loss'] = "{:.4f}".format(background_loss.item())
                else:
                    background_loss = 0
                #latent_loss = 0.
                reconstruction_loss = 0.

            total_losses = reconstruction_loss + \
                background_loss + \
                landmark_loss + \
                emotion_loss + \
                id_loss


            loss_log['sum_non_gan'] = "{:.4f}".format(total_losses.item())

            optimizer.zero_grad()
            total_losses.backward()
            optimizer.step()

            print(epoch, loss_log)

            if iter % 10 == 8:
                if same_face:
                    filename = '{:06}_same_face.png'.format(iter)
                else:
                    filename = '{:06}_{:.2f}_{:.2f}_emo_face.png'.format(iter, src_emotions[0, 0].item(), src_emotions[0, 1].item())

                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                if not os.path.exists(os.path.join(log_path, "personalized_training")):
                    os.makedirs(os.path.join(log_path, "personalized_training"))
                    
                save_image([images[0], generated_images[0]],
                            os.path.join(log_path, "personalized_training", filename), 
                            normalize=True)
                
            iter += 1
    
    person_name = os.path.basename(os.path.normpath(datapath))

    torch.save(stylegan, 
               os.path.join(output_path, 'EmoStyle_{}_{}_{}.pt'.format(inversion_type, ITER_NUM, person_name)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Personalized training script")

    parser.add_argument("--datapath", type=str, default="experiments/personalized_single_4/")
    parser.add_argument("--stylegan2_checkpoint_path", type=str, default="pretrained/ffhq2.pkl")
    parser.add_argument("--emo_mapping_checkpoint_path", type=str, default="checkpoints/emo_mapping_wplus/emo_mapping_wplus_2.pt")
    parser.add_argument("--vggface2_checkpoint_path", type=str, default="pretrained/resnet50_ft_weight.pkl")
    parser.add_argument("--emonet_checkpoint_path", type=str, default="pretrained/emonet_8.pth")
    parser.add_argument("--log_path", type=str, default="logs/personalized")
    parser.add_argument("--inversion_type", type=str, default="e4e")
    parser.add_argument("--output_path", type=str, default="checkpoints/")
    parser.add_argument("--wplus", type=bool, default=True)

    args = parser.parse_args()

    train(
        datapath=args.datapath,
        stylegan2_checkpoint_path=args.stylegan2_checkpoint_path,
        emo_mapping_checkpoint_path=args.emo_mapping_checkpoint_path,
        vggface2_checkpoint_path=args.vggface2_checkpoint_path,
        emonet_checkpoint_path=args.emonet_checkpoint_path,
        log_path=args.log_path,
        inversion_type=args.inversion_type,
        output_path=args.output_path,
        wplus=args.wplus
    )