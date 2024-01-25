"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """
 
from argparse import Namespace
from pathlib import Path

from tqdm import tqdm

from third_party.e4e.psp import pSp

import torch
import torch.nn.functional as F
from torchvision import transforms

class InversionEncoder:
    def __init__(self, checkpoint_path: Path, device: torch.device):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = checkpoint_path
        opts = Namespace(**opts)
        encoder = pSp(opts)

        self.device = device
        self.encoder = encoder.eval().to(device).requires_grad_(False)

    def invert(self, img):
        transform = transforms.Compose([
            transforms.Resize(1024),
            transforms.CenterCrop((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        img = transform(img)
        enc_in = torch.clamp(F.interpolate(img, 256, mode='bicubic'), -1, 1).to(self.device)
            
        out_img, w = self.encoder(enc_in,
                                    randomize_noise=False, return_latents=True,
                                    resize=False, input_code=False)
        return out_img, w
