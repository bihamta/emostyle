import numpy as np
import pickle
import torch
import torch.nn.functional as F

from torch import nn
from .base.resnet import resnet50

class VGGFace2(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()

        self.mean_bgr = torch.FloatTensor([91.4953, 103.8827, 131.0912])

        resnet = resnet50(num_classes=8631)
        with open(checkpoint_path, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        resnet.load_state_dict(weights)
        resnet.eval()

        self.model = nn.Sequential(*list(resnet.children())[:-1])

    def preprocess(self, image):
        image = 255 * image

        min_x = int(0.1 * image.shape[-2])
        max_x = int(0.9 * image.shape[-2])
        min_y = int(0.1 * image.shape[-1])
        max_y = int(0.9 * image.shape[-1])

        image = image[:, :, min_x:max_x, min_y:max_y]
        image = F.interpolate(image, (256, 256))
        # image = transforms.Resize(256)(image)
        # image = transforms.CenterCrop(224)(image)

        start = (256 - 224) // 2
        image = image[:, :, start:224+start, start:224+start]

        image = image[:, [2, 1, 0], :, :] - self.mean_bgr.view(1, -1, 1, 1).to(image.device)

        return image

    def forward(self, image):
        preprocessed = self.preprocess(image)
        embedding = self.model(preprocessed)
        embedding = embedding.view(embedding.size(0), -1)

        embedding = F.normalize(embedding, dim=-1)
        # TODO: l2_normalize embedding

        return embedding