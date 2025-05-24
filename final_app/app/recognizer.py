import numpy as np
from PIL import Image

# app/recognizer.py
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


##import timm  # si usas ViT



class ResNetExtractor(nn.Module):
    def __init__(self, k_freeze=3, pretrained=True, arch='resnet18'):
        """
        k_freeze = nº de *bloques* iniciales a congelar (0 = nada).
        En ResNet el orden es: stem, layer1, layer2, layer3, layer4.
        """
        super().__init__()
        if arch == 'resnet18':
            net = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        elif arch == 'resnet34':
            net = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        elif arch == 'resnet50':
            net = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        elif arch == 'resnet101':
            net = models.resnet101(weights="IMAGENET1K_V1" if pretrained else None)
        elif arch == 'resnet152':
            net = models.resnet152(weights="IMAGENET1K_V1" if pretrained else None)
        self.body = nn.Sequential(*list(net.children())[:-1])   # sin FC
        self.out_dim = net.fc.in_features

        # congela primeros k bloques
        for blk in list(self.body.children())[:k_freeze]:
            blk.requires_grad_(False)

    def forward(self, x):
        feat = self.body(x).flatten(1)    # [B, out_dim]
        return F.normalize(feat, p=2, dim=1)        # [B, out_dim]


class EmbeddingHead(nn.Module):
    def __init__(self, extractor: nn.Module, emb_dim: int = 128):
        super().__init__()
        self.extractor = extractor
        self.proj = nn.Linear(extractor.out_dim, emb_dim)
    def forward(self, x):
        feats = self.extractor(x)
        emb = self.proj(feats)
        return F.normalize(emb, p=2, dim=1)
    

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DummyRecognizer:
    def encode(self, img: Image.Image) -> np.ndarray:
        return np.random.rand(128)


class CowRecognizer:
    """
    `.encode(img_pil) -> np.ndarray de tamaño 128`
    """
    def __init__(self, weights):
  

        # 1) reconstruye arquitectura EXACTAMENTE como en entrenamiento
        ext  = ResNetExtractor(k_freeze=3, arch="resnet18").to(DEVICE)
        self.model = EmbeddingHead(ext, emb_dim=128).to(DEVICE)
        # 2) carga pesos y eval
        self.model.load_state_dict(torch.load(weights, map_location=DEVICE))
        #self.model.load_state_dict(torch.load(weights), map_location=DEVICE)
        #self.model.eval()

        # # 3) misma transformación que en `eval_tfms["resnet"]`
        # self.tf = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=(0.485, 0.456, 0.406),
        #         std=(0.229, 0.224, 0.225)
        #     ),
        # ])

    @torch.no_grad()
    def encode(self, img_pil):
        """
        img_pil: PIL.Image (RGB)
        return : np.float32[128]
        """
        x = self.tf(img_pil).unsqueeze(0).to(DEVICE)   # [1,3,H,W]
        emb = self.model(x).cpu().numpy()[0]           # [128]
        return emb.astype(np.float32)