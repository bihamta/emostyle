from argparse import Namespace
from tqdm import tqdm

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import dnnlib
import os

from models.e4e.psp import pSp
from models.stylegan2_interface import StyleGAN2

class Inversion:
    def __init__(self, latent_path, inversion_type='e4e', cache_only=False, device='cuda') -> None:
        self.latent_path = latent_path
        self.inversion_type = inversion_type
        self.device = device
        self.latents = {}
        self.first_inv_steps = 450
        self.first_inv_lr = 5e-3
        if cache_only == False:
            if self.inversion_type != 'project':
                if self.inversion_type == 'w_encoder':
                    checkpoint_path = 'pretrained/faces_w_encoder.pt'
                elif self.inversion_type == 'e4e':
                    checkpoint_path = 'pretrained/e4e_ffhq_encode.pt'
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                opts = ckpt['opts']
                opts['device'] = self.device
                opts['checkpoint_path'] = checkpoint_path
                opts = Namespace(**opts)
                self.encoder = pSp(opts).eval().to(self.device).requires_grad_(False)

            elif self.inversion_type == 'project':
                self.stylegan = StyleGAN2(
                    checkpoint_path='pretrained/ffhq2.pkl', # 'pretrained/stylegan2-ffhq-config-f.pt'
                    stylegan_size=1024,
                    is_dnn=True,
                    is_pkl=True
                )
                self.stylegan.eval().requires_grad_(False).to(self.device)

    def invert(self, image, image_name):    
        latent = self.calc_inversion(image)
        torch.save(latent, os.path.join(self.latent_path, f'{image_name}_{self.inversion_type}.pt'))

        return latent


    def load_latent(self, image_name):
        if image_name in self.latents:
            return self.latents[image_name]

        # if self.inversion_type == 'e4e':
        #     w_potential_path = f'{self.latent_path}/e4e/{image_name}/0.pt'
        # elif self.inversion_type == 'project':
        #     w_potential_path = f'{self.latent_path}/project/{image_name}/0.pt'
        # elif self.inversion_type == 'w_encoder':
        #     w_potential_path = f'{self.latent_path}/w_encoder/{image_name}/0.pt'

        if not os.path.isfile(os.path.join(self.latent_path, f'{image_name}_{self.inversion_type}.pt')):
            return None
        w = torch.load(os.path.join(self.latent_path, f'{image_name}_{self.inversion_type}.pt'), map_location='cpu')#.to(self.device)
        self.latents[image_name] = w
        return w


    def calc_inversion(self, image):
        if self.inversion_type == 'e4e':
            w = self.get_e4e_inversion(image)

        elif self.inversion_type == 'project':
            # id_image = torch.squeeze((image.to(self.device) + 1) / 2) * 255
            id_image = torch.squeeze((image + 1) / 2) * 255
            w = self.project(id_image, w_avg_samples=600,
                num_steps=self.first_inv_steps,
            )

        elif self.inversion_type == 'w_encoder':
            w = self.get_w_encoder_inversion(image)

        return w


    def get_e4e_inversion(self, image):
        image = (image + 1.) / 2.
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
        image = transform(image).to(self.device)
        _, w = self.encoder(
            image.unsqueeze(0),
            randomize_noise=False,
            return_latents=True,
            resize=False,
            input_code=False
        )
        return w

    def get_w_encoder_inversion(self, image):
        image = (image + 1.) / 2.
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
        image = transform(image).to(self.device)
        _, w = self.encoder(
            image.unsqueeze(0),
            randomize_noise=False,
            return_latents=True,
            resize=False,
            input_code=False
        )
        return w


    def project(
        self,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        use_wandb=False,
        initial_w=None,
        image_log_step=100
    ):
        assert target.shape == (3, self.stylegan.size, self.stylegan.size)

        def logprint(*args):
            if verbose:
                print(*args)

        # Compute w stats.
        logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        z_samples = np.random.RandomState(123).randn(w_avg_samples, 512)
        w_samples = self.stylegan.generate_latent_from_noise(torch.from_numpy(z_samples).to(self.device))  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
        w_avg_tensor = torch.from_numpy(w_avg).to(self.device)
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

        start_w = initial_w if initial_w is not None else w_avg

        # Setup noise inputs.
        noise_bufs = {name: buf for (name, buf) in self.stylegan.generator.named_buffers() if 'noise_const' in name}

        # Load VGG16 feature detector.
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(self.device)

        # Features for target image.
        target_images = target.unsqueeze(0).to(self.device).to(torch.float32)
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        target_features = vgg16(target_images, resize_images=False, return_lpips=True)

        w_opt = torch.tensor(start_w, dtype=torch.float32, device=self.device,
                            requires_grad=True)  # pylint: disable=not-callable
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                    lr=self.first_inv_lr)

        # Init noise.
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        for step in tqdm(range(num_steps)):

            # Learning rate schedule.
            t = step / num_steps
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = (w_opt + w_noise).repeat([1, self.stylegan.num_styles, 1])
            synth_images = self.stylegan.generate(ws)

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images = (synth_images + 1) * (255 / 2)
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

            # Features for synth images.
            synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
            dist = (target_features - synth_features).square().sum()

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss = dist + reg_loss * regularize_noise_weight

            # if step % image_log_step == 0:
            #     with torch.no_grad():
            #         if use_wandb:
            #             global_config.training_step += 1
            #             wandb.log({f'first projection _{w_name}': loss.detach().cpu()}, step=global_config.training_step)
            #             log_utils.log_image_from_w(w_opt.repeat([1, G.mapping.num_ws, 1]), G, w_name)

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        return w_opt.repeat([1, 18, 1])


if __name__ == '__main__':
    inversion = Inversion(
        latent_path='experiments/expr2_1024/latents',
        inversion_type='project',
        device='cuda'
    )

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    from PIL import Image
    image = Image.open('face_augmented_79.png').convert('RGB')
    image = transform(image)

    latent = inversion.invert(image, 'face_augmented_79.png')

    # checkpoint_path = 'pretrained/stylegan2-ffhq-config-f.pt'
    # ckpt = torch.load(checkpoint_path)
    # generator = Generator(1024, 512).eval().requires_grad_(False).to('cuda')
    # generator.load_state_dict(ckpt["g"])
    # synthesis = generator(latent[0], randomize_noise=False).squeeze()

    import pickle
    try:
        with dnnlib.util.open_url(str('pretrained/ffhq2.pkl')) as f:
            G = pickle.load(f)['G_ema'].synthesis
    except Exception as e:
        G = torch.load('pretrained/ffhq2.pkl')
    G = G.cuda()

    synthesis = G(latent, noise_mode='const', force_fp32=True).squeeze()

    image = (image.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    synthesis = (synthesis.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

    Image.fromarray(image, 'RGB').save('orig.png')
    Image.fromarray(synthesis, 'RGB').save('inverted.png')