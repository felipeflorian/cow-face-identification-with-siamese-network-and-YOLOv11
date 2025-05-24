# Inferencia de a imagenes
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torchvision import models
import torch.nn as nn
from .recognizer import ResNetExtractor, EmbeddingHead



### Función para dada una imagen de vaca y un modelo, devolver la etiqueta de la vaca a la que pertenece
def predict_image(image_tensor, model, gallery_embs, gallery_labels, device, threshold: float = None):
    """
    Clasifica una sola imagen:
      - image_tensor: torch.Tensor [C, H, W], NO en batch.
      - model:  embedder.
      - gallery_embs: torch.Tensor [G, D], precomputado.
      - gallery_labels: np.array [G]
      - threshold: distancia máxima para aceptar; si None → closed-set puro.

    Devuelve la etiqueta predicha o la cadena "unknown".
    """
    model.eval()
    # 1) Prepara image_tensor como batch=1
    img = image_tensor.unsqueeze(0).to(device)            # [1, C, H, W]
    with torch.no_grad():
        emb = F.normalize(model(img), dim=1).cpu()         # [1, D]

    # 2) Calcula distancias L2 vs galería
    #    torch.cdist permite hasta 2D: (1, D) vs (G, D) → (1, G)
    dists = torch.cdist(emb, gallery_embs, p=2).squeeze(0) # [G]

    # 3) Encuentra el vecino más cercano
    min_dist, idx = torch.min(dists, dim=0)
    pred_label = gallery_labels[idx.item()]

    # 4) Open-set: rechazo si supera umbral

    print("Treshold:", threshold)
    if threshold is not None and min_dist.item() > threshold:
        return "unknown", min_dist.item()

    # 5) Closed-set o conocido
    return pred_label, min_dist.item()

