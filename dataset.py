import os
import glob
import numpy as np
import pickle
from PIL import Image
from invert import Inversion
import torch.utils.data
import torchvision.transforms as transforms

def default_image_loader(path):
    return Image.open(path).convert('RGB') #.transpose(0, 2, 1)

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, datapath='./dataset', mode='training', loader=default_image_loader):
        self.datapath = datapath
        self.mode = mode
        self.loader = loader
        self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        image_path = os.path.join(self.datapath, f'{index:06}.png')
        latent_path = os.path.join(self.datapath, f'{index:06}.npy')

        image = self.loader(image_path)
        latent = np.load(latent_path, allow_pickle=False)

        if (self.transform):
            image = self.transform(image)
            latent = torch.from_numpy(latent).float()

        return image, latent

    def __len__(self):
        return 1000

class PersonalizedSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, datapath='./dataset', inversion_type='e4e', loader=default_image_loader):
        self.datapath = datapath
        self.loader = loader
        self.inversion_type = inversion_type
        # self.transform = transforms.Compose([
        #         transforms.Resize((256, 256)),
        #         transforms.ToTensor()
        #     ])
        self.image_paths = list(glob.glob(os.path.join(self.datapath, 'images/*')))
        self.transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
        self.inversion = Inversion(
                latent_path=os.path.join(self.datapath, 'latents'),
                inversion_type=self.inversion_type,
                cache_only=True,
                device='cpu'
        )

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.loader(image_path)

        if (self.transform):
            image = self.transform(image)
        image = (image + 1.) / 2.
        latent = self.inversion.load_latent(os.path.basename(image_path).split('.')[0])
        latent = latent.squeeze()
        
        return image, latent

    def __len__(self):
        return len(self.image_paths)
