import math
import pickle
import dnnlib
import torch
import torch.nn as nn

from models.stylegan2 import Generator, Mapping


class StyleGAN2(nn.Module):
    def __init__(self, checkpoint_path, stylegan_size, is_dnn=False, is_pkl=True) -> None:
        super().__init__()
        self.size = stylegan_size
        self.num_styles = int(math.log(self.size, 2)) * 2 - 2
        self.is_pkl = is_pkl
        self.is_dnn = is_dnn

        if self.is_dnn:
            if self.is_pkl:
                with dnnlib.util.open_url(str(checkpoint_path)) as f:
                    G = pickle.load(f)['G_ema'].float()
                    self.generator = G.synthesis
                    self.mapping = G.mapping    
            else:
                with open(checkpoint_path, 'rb') as f:
                    new_G = torch.load(f).float()
                    self.generator = new_G.generator
                    self.mapping = new_G.mapping
        else:
            ckpt = torch.load(checkpoint_path)
            self.mapping = Mapping(stylegan_size, 512, 8)
            self.generator = Generator(stylegan_size, 512)

            self.mapping.load_state_dict(ckpt["m"])
            self.generator.load_state_dict(ckpt["g"])
            self.latent_avg = ckpt["latent_avg"]

    def generate_latent_from_noise(self, z_samples):
        if self.is_dnn:
            latents = self.mapping(z_samples, None)
        else:
            latents = self.mapping(z_samples)
        return latents

    def generate_from_noise(self, z_samples, truncation_psi=0.7):
        if self.is_dnn:
            latents = self.mapping(z_samples, None, truncation_psi=truncation_psi)
            images = self.generator(
                latents,
                noise_mode='const',
                force_fp32=True
            )

        else:
            latents = self.mapping(z_samples)
            images = self.generator(
                [latents],
                truncation=truncation_psi,
                truncation_latent=self.latent_avg,
                randomize_noise=False,
            )

        return latents, images


    def generate(self, latents):
        if (self.is_dnn):
            if (len(latents.shape) == 2):
                latents = latents.unsqueeze(1).repeat(1, 18, 1)
            images = self.generator(
                latents,
                noise_mode='const',
                force_fp32=True
            )

        else:
            images = self.generator(
                [latents]
            )

        return images