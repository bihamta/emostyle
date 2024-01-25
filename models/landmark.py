import os
import sys
import torch
import torch.nn as nn

from urllib.parse import urlparse
from torch.hub import download_url_to_file, HASH_REGEX
try:
    from torch.hub import get_dir
except BaseException:
    from torch.hub import _get_torch_home as get_dir

default_model_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-4a694010b9.zip',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-6c4283c0e0.zip',
}

models_urls = {
    '1.6': {
        '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.6-c827573f02.zip',
        '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.6-ec5cf40a1d.zip',
        'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth_1.6-2aa3f18772.zip',
    },
    '1.5': {
        '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.5-a60332318a.zip',
        '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.5-176570af4d.zip',
        'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth_1.5-bc10f98e39.zip',
    },
}

# Pytorch load supports only pytorch models
def load_file_from_url(url, model_dir=None, progress=True, check_hash=False, file_name=None):
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'pretrained')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file

class FaceAlignment(nn.Module):
    def __init__(self):
        super().__init__()

        # Get the face detector
        # face_detector_module = __import__('face_alignment.detection.' + face_detector,
        #                                 globals(), locals(), [face_detector], 0)
        # face_detector_kwargs = face_detector_kwargs or {}
        # self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose, **face_detector_kwargs)
    
        network_name = '2DFAN-4'
        self.face_alignment_net = torch.jit.load(
            load_file_from_url(models_urls.get("1.6", default_model_urls)[network_name]))

        self.face_alignment_net.eval()

    def heatmap2poses(self, heatmaps, scale = 1):
        assert heatmaps.dim() == 4, 'Score maps should be 4-dim (B, nJoints, H, W)'
        maxval, idx = torch.max(heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1), 2)

        maxval = maxval.view(heatmaps.size(0), heatmaps.size(1), 1)
        idx = idx.view(heatmaps.size(0), heatmaps.size(1), 1)

        preds = idx.repeat(1, 1, 2).float()  # (B, njoint, 2)

        preds[:, :, 0] = (preds[:, :, 0]) % heatmaps.size(3)  # + 1
        preds[:, :, 1] = torch.floor((preds[:, :, 1]) / heatmaps.size(3))  # + 1

        pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
        preds *= pred_mask
        return (preds * scale), maxval

    def forward(self, image):
        heatmaps = self.face_alignment_net(image)
        scale = image.shape[-1] / heatmaps.shape[-1]
        preds, maxval = self.heatmap2poses(heatmaps, scale)
        return preds, maxval