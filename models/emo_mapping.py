import math
import torch
import torch.nn.functional as F
from torch import nn
from .stylegan2 import EqualLinear

class EmoMappingV1(nn.Module):
    def __init__(
        self,
        latent_dim,
        n_mlp,
        lr_mlp=0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        layers = []
        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.latent_net = nn.Sequential(*layers)

    def forward(self, z):
        return self.latent_net

class EmoMappingWplus(nn.Module):
    def __init__(
        self,
        size,
        emotion_dim,
        latent_dim,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.size = size
        self.num_styles = int(math.log(self.size, 2)) * 2 - 2
        self.emotion_enc = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, emotion_dim)
        )
        
        self.latent_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emotion_dim + latent_dim, 2048),
                nn.LeakyReLU(0.2),
                nn.Linear(2048, 1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, self.latent_dim)
            ) for _ in range(self.num_styles)
        ])
        # self.latent_net.apply(self.init_params)

    def init_params(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, latents, emotions):
        emotion_emb = self.emotion_enc(emotions)
        output_latents = []
        for i, style_net in enumerate(self.latent_net):
            output_latents.append(
                style_net(torch.cat([latents[:, i, :], emotion_emb], dim=1))
            )
        
        return torch.stack(output_latents, dim=1)

class EmoMappingW(nn.Module):
    def __init__(
        self,
        emotion_dim,
        latent_dim,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.emotion_enc = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, emotion_dim)
        )

        self.latent_net = nn.Sequential(
            nn.Linear(emotion_dim + latent_dim, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.latent_dim)
        )
        # self.latent_net.apply(self.init_params)

    def init_params(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, latents, emotions):
        emotion_emb = self.emotion_enc(emotions)
        return self.latent_net(torch.cat([latents, emotion_emb], dim=1))

class EmoMappingDiscriminator(nn.Module):
    def __init__(
        self,
        latent_dim,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.discriminator_net = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, latent):
        return self.discriminator_net(latent)