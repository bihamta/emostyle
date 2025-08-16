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


def test(
    images_path: str,
    stylegan2_checkpoint_path: str,
    checkpoint_path: str,
    output_path: str,
    test_mode: str,
    valence: list,
    arousal: list,
    wplus: bool
):
    stylegan = StyleGAN2(
        checkpoint_path=stylegan2_checkpoint_path,
        stylegan_size=INPUT_SIZE,
        is_dnn=True,
        is_pkl=True
    )
    stylegan.eval()
    stylegan.requires_grad_(False)

    ckpt_emo = torch.load("pretrained/emonet_8.pth")
    ckpt_emo = {k.replace('module.', ''): v for k, v in ckpt_emo.items()}
    emonet = EmoNet(n_expression=8)
    emonet.load_state_dict(ckpt_emo, strict=False)
    emonet.eval()

    ckpt_emo_mapping = torch.load(
        checkpoint_path, map_location=torch.device('cpu'))
    if wplus:
        emo_mapping = EmoMappingWplus(INPUT_SIZE, EMO_EMBED, STG_EMBED)
    else:
        emo_mapping = EmoMappingW(EMO_EMBED, STG_EMBED)
   # emo_mapping = EmoMappingW(EMO_EMBED, STG_EMBED)
    emo_mapping.load_state_dict(ckpt_emo_mapping['emo_mapping_state_dict'])
    emo_mapping.eval()

    if is_cuda:
        emo_mapping.cuda()
        stylegan.cuda()
        emonet.cuda()

    latents = {}

    if test_mode == 'random':
        random_images = random.sample(range(1, 70000), 100)
        with open('random.pkl', 'wb') as f:
            pickle.dump(random_images, f)

        for image_number in range(len(random_images)):
            latent_path = images_path + \
                str(random_images[image_number]).zfill(6) + ".npy"
            image_path = images_path + \
                str(random_images[image_number]).zfill(6) + ".png"
            image_latent = np.load(latent_path, allow_pickle=False)
            if wplus:
                image_latent = np.expand_dims(image_latent[:, :], 0)
            else:
                image_latent = np.expand_dims(image_latent[0, :], 0)
            image_latent = torch.from_numpy(image_latent).float()
            latents[image_path] = image_latent

    elif test_mode == 'folder_images':
        #         for image_file in glob.glob(images_path+"*"):
        #            # latent_path = "1.npy"
        #             latent_path = os.path.splitext(image_file)[0]+'.npy'
        #             image_latent = np.load(latent_path, allow_pickle=False)
        #             if wplus:
        #                 image_latent = np.expand_dims(image_latent[:, :], 0)
        #             else:
        #                 image_latent = np.expand_dims(image_latent[0, :], 0)
        #             image_latent = torch.from_numpy(image_latent).float()
        #             latents[image_file] = image_latent
        for image_file in glob.glob(images_path+"*"):
            latent_path = os.path.splitext(image_file)[0]+'.npy'
            image_path = os.path.splitext(image_file)[0]+'.png'
            image_latent = np.load(latent_path, allow_pickle=False)
            if wplus:
                image_latent = np.expand_dims(image_latent[:, :], 0)
            else:
                image_latent = np.expand_dims(image_latent[0, :], 0)
            image_latent = torch.from_numpy(image_latent).float()
            latents[image_path] = image_latent

    emos_data = {}
    for v in range(len(valence)):
        for a in range(len(arousal)):
            emos_data[(valence[v], arousal[a])] = []

    num_images = len(valence) * len(arousal)
    for img, latent in latents.items():
        input_image = Image.open(img).convert('RGB')  # .transpose(0, 2, 1)
        image_name = os.path.basename(img)

        _, ax_g = plt.subplots(1, num_images, figsize=(100, 50))
        plt.subplots_adjust(left=.05, right=.95, wspace=0, hspace=0)
        iter = 0

        image_tensors = []
        for v_idx in tqdm(range(len(valence))):
            for a_idx in range(len(arousal)):
                emotion = torch.FloatTensor(
                    [valence[v_idx], arousal[a_idx]]).float().unsqueeze_(0)

                # If i don't add it, i will get an RuntimeError.
                # Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
                emotion = emotion.to(device=torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'))
                latent = latent.to(device=torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'))
                fake_latents = latent + emo_mapping(latent, emotion)
                generated_image_tensor = stylegan.generate(fake_latents)
                generated_image_tensor = (generated_image_tensor + 1.) / 2.
                emo_embed = emonet(generated_image_tensor)
                emos = (emo_embed[0][0], emo_embed[0][1])

                generated_image = generated_image_tensor.detach().cpu().squeeze().numpy()
                generated_image = np.clip(
                    generated_image*255, 0, 255).astype(np.int32)
                generated_image = generated_image.transpose(
                    1, 2, 0).astype(np.uint8)

                # emos_data[(valence[v_idx], arousal[a_idx])].append(
                #     abs(emos[0] - valence[v_idx]), abs(emos[1] - arousal[a_idx]))
                # add []
                emos_data[(valence[v_idx], arousal[a_idx])].append(
                    [abs(emos[0] - valence[v_idx]), abs(emos[1] - arousal[a_idx])])
                image_tensors.append(generated_image_tensor)

                ax_g[iter].imshow(generated_image)
                ax_g[iter].set_title("V: p{:.2f}, r{:.2f}, A: p{:.2f}, r{:.2f}".format(
                    valence[v_idx], emos[0], arousal[a_idx], emos[1]), fontsize=40)
                ax_g[iter].axis('off')
                iter += 1

        plt.savefig("result.png", bbox_inches='tight')

        fig = plt.figure(figsize=(80, 20))

        with open('emos_data.pkl', 'wb') as f:
            pickle.dump(emos_data, f)
        output_image = Image.open("result.png").convert('RGB')
        grid = plt.GridSpec(1, num_images+1, wspace=0.05, hspace=0)
        ax_output = fig.add_subplot(grid[0, 1:])
        ax_input = fig.add_subplot(grid[0, 0])
        ax_input.imshow(input_image)
        ax_input.axis('off')
        ax_output.imshow(output_image)
        ax_output.axis('off')
        pos_old = ax_input.get_position()
        pos_new = [pos_old.x0,  0.335,  pos_old.width, pos_old.height]
        ax_input.set_position(pos_new)
        plt.savefig(output_path + "result_{}".format(image_name),
                    bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script")

    parser.add_argument("--images_path", type=str, default="dataset/1024_pkl/")
    parser.add_argument("--stylegan2_checkpoint_path",
                        type=str, default="pretrained/ffhq2.pkl")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoints/emo_mapping_w.pt")
    parser.add_argument("--output_path", type=str, default="results/")
    parser.add_argument("--test_mode", type=str, default="random")
    parser.add_argument("--valence", type=float, nargs='+', default=[0.5])
    parser.add_argument("--arousal", type=float, nargs='+', default=[0.5])
    parser.add_argument("--wplus", type=bool, default=False)

    args = parser.parse_args()

    test(
        images_path=args.images_path,
        stylegan2_checkpoint_path=args.stylegan2_checkpoint_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        test_mode=args.test_mode,
        valence=args.valence,
        arousal=args.arousal,
        wplus=args.wplus
    )
